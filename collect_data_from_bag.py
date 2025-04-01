# encoding: utf-8
#!/usr/bin/env python

import argparse
import csv
# from dbw_mkz_msgs.msg import *
import fastbag
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import rosbag
from scipy import interpolate
import sys
from pluspy.topic_utils import MessageDecoder
from google.protobuf.pyext._message import RepeatedCompositeContainer
from scipy.interpolate import interp1d
import datetime
from generate_data_utils import *
from verify_utils import *
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def data_interpolation(x_original, y_original, x_new):
    dist_for_interp = 0.1  # Any number > 0 would suffice.
    if (not isinstance(x_original, (list, np.ndarray))
            or not isinstance(y_original, (list, np.ndarray))
            or len(x_original) != len(y_original)
            or len(x_original) < 2):
        raise RuntimeError("Error using data_interpolation(): input lists must have same length which is >= 2!")
    if not all([x_original[i] > x_original[i - 1] for i in range(1, len(x_original))]):
        raise RuntimeError("Error using data_interpolation(): input x_original must be in assending order with no duplicates!")

    x_extend_left = min(min(x_new), x_original[0]) - dist_for_interp
    x_extend_right = max(max(x_new), x_original[-1]) + dist_for_interp

    y_extend_left = (y_original[1] - y_original[0]) / (x_original[1] - x_original[0]) * (x_extend_left - x_original[0])
    y_extend_left += y_original[0]
    y_extend_right = (y_original[-1] - y_original[-2]) / (x_original[-1] - x_original[-2]) * (x_extend_right - x_original[-1])
    y_extend_right += y_original[-1]

    x_original = np.insert(x_original, 0, x_extend_left)
    x_original = np.append(x_original, x_extend_right)
    y_original = np.insert(y_original, 0, y_extend_left)
    y_original = np.append(y_original, y_extend_right)

    data_interpolate = interpolate.interp1d(x_original, y_original)

    return list(data_interpolate(x_new))

def set_order_by_x(x_data, y_data):
    if (not isinstance(x_data, (list, np.ndarray))
            or not isinstance(y_data, (list, np.ndarray))
            or len(x_data) != len(y_data)
            or len(x_data) < 2):
        raise RuntimeError("Error using set_order_by_x(): input lists must have same length which is >= 2!")
    data_list = [(x, y) for x, y in zip(x_data, y_data)]

    def get_first_ele(input_list):
        return input_list[0]
    data_list.sort(key=get_first_ele)
    return [p[0] for p in data_list], [p[1] for p in data_list]

def remove_duplicates_by_x(x_data, y_data):
    if (not isinstance(x_data, (list, np.ndarray))
            or not isinstance(y_data, (list, np.ndarray))
            or len(x_data) != len(y_data)
            or len(x_data) < 2):
        raise RuntimeError("Error using data_interpolation(): input lists must have same length which is >= 2!")
    if not all([x_data[i] >= x_data[i - 1] for i in range(1, len(x_data))]):
        raise RuntimeError("Error using data_interpolation(): input x_original must be in assending order!")

    output_pairs = []
    for x, y in zip(x_data, y_data):
        if not output_pairs or output_pairs[-1][0] != x:
            output_pairs.append((x, y))

    return [p[0] for p in output_pairs], [p[1] for p in output_pairs]

class FileFinder:
    def __init__(self, file_suffix_list):
        self._baglist = []
        self._file_suffix_list = file_suffix_list

    def append_filename_to_baglist(self, path, file_name):
        if (file_name.split('.'))[-1] in self._file_suffix_list:
            self._baglist.append(os.path.abspath(os.path.join(path, file_name)))

    def find_bags_in_(self, dir_name_list):
        for dir_name in dir_name_list:
            if os.path.isfile(dir_name):
                self.append_filename_to_baglist('', os.path.abspath(dir_name))
            elif os.path.isdir(dir_name):
                for path, dirs, files in os.walk(dir_name):
                    for file_name in files:
                        self.append_filename_to_baglist(path, file_name)
        self._baglist = list(set(self._baglist))
        return self._baglist

class BagReader():
    def __init__(self, file_name):
        self.file_name = os.path.abspath(file_name)
        self.last_time_sec = None
        self.start_time_sec = None
        self.msg_decoder = MessageDecoder()
        self.interested_topics = []
        self.extracted_data_dict = {}
        self.bag_timestamp_list = []

    def read_bag(self, args):
        # check if input filename is valid.
        if not os.path.isfile(self.file_name):
            raise ValueError('input filename is not a valid file: {}'.format(self.file_name))

        # validate topics.
        if not isinstance(args.topics, list) or not args.topics:
            raise ValueError('empty topic input, or input topic not being a list: {}'.format(args.topics))
        self.interested_topics = args.topics
        logger.info("")
        logger.info("the following topic will be extracted: {}".format(self.interested_topics))

        # Load data.
        logger.info('')
        logger.info('Opening bag file: {}'.format(self.file_name))
        try:
            if self.file_name.endswith(".bag"):
                bag = rosbag.Bag(self.file_name)
            elif self.file_name.endswith(".db"):
                bag = fastbag.Reader(self.file_name)
                bag.open()
            else:
                raise IOError("invalid bagfile suffix!")
        except IOError as e:
            raise IOError('Unable to read bag: {}, {}'.format(self.file_name, e))

        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        logger.info('Reading {} messages (duration: {:.1f} sec)...'.format(
            bag.get_message_count(), bag_end_time - bag_start_time))
        logger.info('Starting time {:.0f}'.format(bag_start_time))
        logger.info('Ending time {:.0f}'.format(bag_end_time))

        # Record data
        extracted_data_dict = {topic: [] for topic in self.interested_topics}
        self.bag_timestamp_list = []
        for msg_topic, msg, t in bag.read_messages():
            if msg_topic not in self.interested_topics:
                continue
            # read protobuf message
            message = self.msg_decoder.decode(msg_topic, msg, raw=False)
            if msg_topic in args.show_topic:
                logger.info(message)

            self.bag_timestamp_list.append(t.to_sec())
            for topic, data_list in extracted_data_dict.items():
                data_list.append(message if topic == msg_topic else None)

        logger.info('Finished reading: {}'.format(self.file_name))

        data_lengths = list(set([len(data_list) for data_list in extracted_data_dict.values()]))
        if len(data_lengths) != 1 or data_lengths[0] != len(self.bag_timestamp_list):
            raise RuntimeError("extracted_data_dict should have same lengths! but got: {}, {}"
                               .format(data_lengths, len(self.bag_timestamp_list)))

        if data_lengths == [0]:
            raise RuntimeError("read no valid data from bag file")

        self.extracted_data_dict = extracted_data_dict

    def to_csv(self, output_folder):
        output_file = '.'.join(os.path.basename(self.file_name).split('.')[:-1]) + '_data.csv'
        output_file = os.path.join(os.path.abspath(output_folder), output_file)
        logger.info('')
        logger.info('Start saving as csv: {}'.format(output_file))

        with open(output_file, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            spamwriter.writerow(['bag_time'] + self.interested_topics)
            for idx in range(len(self.bag_timestamp_list)):
                row = [self.bag_timestamp_list[idx]]
                for topic in self.interested_topics:
                    msg = self.extracted_data_dict[topic][idx]
                    row.append(self._get_message_dict(msg) if msg is not None else {})
                spamwriter.writerow(row)

        logger.info('Successfully finished saving as csv: {}'.format(output_file))

    @staticmethod
    def _get_primary_attri(msg_str):
        attri_names = msg_str.splitlines()
        attri_names = [line.strip() for line in attri_names if not line.startswith(' ')]
        attri_names = [line.split(':')[0].split('{')[0].split('}')[0].strip() for line in attri_names]
        attri_names = [line for line in attri_names if line]
        return attri_names

    def _get_message_dict(self, msg):
        msg_dict = {}
        msg_str = str(msg)
        attri_names = self._get_primary_attri(msg_str)
        for attri_name in attri_names:
            attri_value = getattr(msg, attri_name, None)
            if attri_value is None:
                continue
            if isinstance(attri_value, (float, int, str, bool)):
                msg_dict[attri_name] = attri_value
            elif isinstance(attri_value, (list, RepeatedCompositeContainer)):
                msg_dict[attri_name] = [self._get_message_dict(v) for v in attri_value]
            else:
                msg_dict[attri_name] = self._get_message_dict(attri_value)
        return msg_dict

    @property
    def extracted_data(self):
        return dict(self.extracted_data_dict)

    @property
    def bag_timestamps(self):
        return list(self.bag_timestamp_list)


class BagDataAnalyzer:
    def __init__(self, bag_times, data_dict):
        self.bag_times = []
        self.init_timestamp = np.nan
        self.data_dict = {}
        self.append_data(bag_times, data_dict)

    def append_data(self, incomming_bag_times, incomming_data_dict):
        # data validity check
        self._check_input_data_validity(incomming_bag_times, incomming_data_dict)

        # fill both incomming data and existing data with lists of the same topics
        old_data_length = [len(v) for v in self.data_dict.values()][0] if self.data_dict else 0
        new_data_length = [len(v) for v in incomming_data_dict.values()][0] if incomming_data_dict else 0
        for new_topic in incomming_data_dict.keys():
            if new_topic not in self.data_dict:
                self.data_dict[new_topic] = [None for _ in range(old_data_length)]
        for existing_topic in self.data_dict.keys():
            if existing_topic not in incomming_data_dict:
                incomming_data_dict[existing_topic] = [None for _ in range(new_data_length)]

        # append the incomming data in front or from behind
        if np.isnan(self.init_timestamp) or self.init_timestamp > incomming_bag_times[0]:
            self.init_timestamp = incomming_bag_times[0]
            self.bag_times = incomming_bag_times + self.bag_times
            for topic, existing_msg_list in self.data_dict.items():
                self.data_dict[topic] = incomming_data_dict[topic] + existing_msg_list
        else:
            self.bag_times += incomming_bag_times
            for topic in self.data_dict.keys():
                self.data_dict[topic] += incomming_data_dict[topic]

    def _parse_data_from_string(self, input_string, use_bag_time=False):
        '''
        This function yields two lists which can be directly used as x and y in plotting'''
        data_strings = set(re.findall(r"/[\w/]+:[\w\.]+", input_string))
        if not data_strings:
            raise ValueError("invalid input string for _parse_data_from_string: {}".format(input_string))
        raw_data_dict = {string: [] for string in data_strings}

        # topic of which the timestamp will be used as x axis
        timestamp_topic = ""
        raw_timestamp_list = []
        ordered_timestamp_list = []

        for string in data_strings:
            topic = string.split(":")[0]
            if topic not in self.data_dict:
                raise RuntimeError("need to read the topic before paring it!!!")

            # fill raw_timestamp_list
            if not timestamp_topic:
                timestamp_topic = topic
                raw_timestamp_list = self._get_timestamps(topic, use_bag_time)
                if not raw_timestamp_list:
                    raise RuntimeError("failed to get valid timestamps for topic: {}, when parsing: {}".format(topic, input_string))

                if all([np.isnan(v) for v in raw_timestamp_list]):
                    return [np.nan], [np.nan]

                raw_timestamp_list = [v - self.init_timestamp for v in raw_timestamp_list]
                ordered_timestamp_list = sorted(raw_timestamp_list)


            # fill raw_data_dict
            fields = string.split(":")[1].split(".")
            try:
                for msg in self.data_dict[topic]:
                    if msg is None:
                        continue
                    value = msg
                    for field in fields:
                        if isinstance(value, RepeatedCompositeContainer):
                            raise RuntimeError("plotting RepeatedCompositeContainer is not supported yet! {}".format(input_string))
                        else:
                            value = getattr(value, field)
                    raw_data_dict[string].append(value)
            except KeyError as e:
                raise ValueError("topic not read in data_dict: {}, when reading: {}, {}".format(topic, input_string, e))
            except AttributeError as e:
                logger.error("error when drawing {}: failed to get data: {}, {}".format(input_string, data_strings, e))
                return [np.nan], [np.nan]

            # do interpolation if need to
            if timestamp_topic != topic:
                timestamps_before_interp = self._get_timestamps(topic, use_bag_time)
                if not timestamps_before_interp:
                    raise RuntimeError("failed to get valid timestamps for topic: {}, when parsing: {}".format(topic, input_string))
                timestamps_before_interp = [v - self.init_timestamp for v in timestamps_before_interp]
                data_before_interp = raw_data_dict[string]
                timestamps_before_interp, data_before_interp = set_order_by_x(timestamps_before_interp, data_before_interp)
                timestamps_before_interp, data_before_interp = remove_duplicates_by_x(timestamps_before_interp, data_before_interp)
                raw_data_dict[string] = data_interpolation(timestamps_before_interp, data_before_interp, raw_timestamp_list)

            _, raw_data_dict[string] = set_order_by_x(raw_timestamp_list, raw_data_dict[string])

        # modify exec string
        line_to_exec = input_string
        for string in data_strings:
            line_to_exec = line_to_exec.replace(string, "np.array(raw_data_dict['{}'])".format(string))

        data_for_y_axis = []
        local_vars = {"raw_data_dict": raw_data_dict, "np": np}
        exec("data_for_y_axis = {}".format(line_to_exec), {}, local_vars)
        data_for_y_axis = local_vars["data_for_y_axis"]

        return ordered_timestamp_list, data_for_y_axis

    def _check_input_data_validity(self, incomming_bag_times, incomming_data_dict):
        if (not isinstance(incomming_bag_times, (list, np.ndarray))
                or len(incomming_bag_times) < 1
                or not isinstance(incomming_bag_times[0], float)
                or np.isnan(incomming_bag_times[0])):
            raise ValueError("invalid input for incomming_bag_times!")

        if (not isinstance(incomming_data_dict, dict)
                or not all([isinstance(v, list) for v in incomming_data_dict.values()])
                or len(set([len(v) for v in incomming_data_dict.values()])) != 1):
            raise ValueError("invalid input for incomming_data_dict!")

        if [len(v) for v in incomming_data_dict.values()][0] != len(incomming_bag_times):
            raise ValueError("length of incomming_data_dict and incomming_bag_times does not match! got: {} and {}."
                .format([len(v) for v in incomming_data_dict.values()][0], len(incomming_bag_times)))

    def _get_timestamps(self, topic, use_bag_time):
        msg_list = [d for d in self.data_dict[topic] if d is not None]
                
        if not msg_list:
            logger.error("No valid data to get header timestamps from: {}".format(topic))
            return [np.nan]

        if use_bag_time:
            return [self.bag_times[idx] for idx, v in self.data_dict[topic] if v is not None]

        if hasattr(msg_list[0], 'header') and hasattr(msg_list[0].header, 'envelope_timestamp_msec'):
            return [msg.header.envelope_timestamp_msec * 1e-3 for msg in msg_list]

        if (hasattr(msg_list[0], 'header') and hasattr(msg_list[0].header, 'stamp')
                                           and hasattr(msg_list[0].header.stamp, 'secs')
                                           and hasattr(msg_list[0].header.stamp, 'nsecs')):
            return [msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 for msg in msg_list]

        raise RuntimeError("failed to get header timestamps from topic: {}, example: {}".format(topic, msg_list[0]))

    def sort_topic(self, topic_dict):
        # if different topic timestamp is different, sync to the same
        # select data in auto model
        min_length = num_size
        auto_t_list, flag_list = self._parse_data_from_string(auto_mode_flag_topic)
        
        use_t_list = []
        use_t_list.append([])
        
        for t, flag in zip(auto_t_list, flag_list):
            if flag == 1:
                use_t_list[-1].append(t)
            else:
                if len(use_t_list[-1]) > min_length:
                    use_t_list.append([])
                else:
                    use_t_list.pop()
                    use_t_list.append([])     
       
        if len(use_t_list[-1]) < min_length:
            use_t_list.pop()

        dataset_total = []
        for first_t_list in use_t_list:
            dataset = {}
            for key, value in topic_dict.items():
                t_list, value_list = self._parse_data_from_string(value)

                if first_t_list != t_list:
                    interp_function = interp1d(t_list , value_list, kind='linear',fill_value='extrapolate')
                    value_list = interp_function(first_t_list)

                dataset[key] = value_list

            dataset_total.append(dataset)

        return dataset_total

# need topic from bag
NEED_TOPIC_BAG = {
    "steer": "/vehicle/dbw_reports:steering_report.steering_wheel_angle",
    "xpos_x": "/navsat/odom:pose.pose.position.x",
    "xpos_y": "/navsat/odom:pose.pose.position.y",
    "xpos_z": "/navsat/odom:pose.pose.position.y",

    "xori_x": "/navsat/odom:pose.pose.orientation.x",
    "xori_y": "/navsat/odom:pose.pose.orientation.y",
    "xori_z": "/navsat/odom:pose.pose.orientation.z",
    "xori_w": "/navsat/odom:pose.pose.orientation.w",
    
    "avel_z": "/vehicle/status:yawrate",

    "xacc_x": "/vehicle/status:x",
    "xacc_y": "/vehicle/status:y",
    "xacc_z": "/vehicle/status:z",

    "throttle": "/vehicle/status:acc_fused",
    "xvel_x" : "/vehicle/status:v",
}

auto_mode_flag_topic = "/vehicle/dbw_reports:superpilot_enabled"

# choose vehicle name for save data
# Vehicle_Name = "pdb_c11"
# Vehicle_Name = "pde_a1"
Vehicle_Name = "debug"

# every pkl data step length
num_size = 2000

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bag_reader')
    
    parser.add_argument('input', nargs='+', type=str, help="Abspath to bag or db, or folders containing them. Seperate with spaces ONLY.")
    parser.add_argument('--output', '-o', default=os.path.dirname(os.path.abspath(__file__)), type=str,
                        help="Abspath of folder to put all output files in. {} as default.".format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--show-topic', default=[], type=str, help="Topics to show in terminal / logfile")
    # csv args
    parser.add_argument('--topics', '-t', default=[], nargs='+', type=str, help="Extra topics to extract")
    parser.add_argument('--to-csv', default=False, action='store_true', help="Whether to generate origin data csvfile")

    args = parser.parse_args()
    
    # if Vehicle_Name == "pde_a1":
    #     file_save_path = "/disk/collect_data_from_anycar/data_from_bag/data_use_steer_angle/pde-a1"
    # if Vehicle_Name == "pdb_c11":
    #     file_save_path = "/disk/collect_data_from_anycar/data_from_bag/data_use_steer_angle/pdb-c11"
    # if Vehicle_Name == "debug":
    #     file_save_path = "/disk/collect_data_from_anycar/check_data/verify_bag_data_0310"

    file_save_path = "/disk/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file"

    os.makedirs(file_save_path, exist_ok=True)
    
    input_bag_list = FileFinder(file_suffix_list=['bag', 'db']).find_bags_in_(args.input)
    if not input_bag_list:
        raise argparse.ArgumentError("invalid input argument, no bag or db found in: {}".format(args.input))
    args.input = input_bag_list

    args.output = os.path.abspath(args.output)
    if not os.path.isdir(args.output):
        raise argparse.ArgumentError("invalid output argument, which is not a folder: {}".format(args.output))

    # collect need topic name
    for bag_topic in NEED_TOPIC_BAG.values():
        topic_strings = set(re.findall(r"/[\w/]+:[\w\.]+", bag_topic))
        if topic_strings:
            args.topics += [string.split(":")[0] for string in topic_strings]

    for file_name in args.input:
        bag_reader = BagReader(file_name)
        try:
            bag_reader.read_bag(args)
        except Exception as e:
            print("Error reading bag: {}".format(file_name))
            print(e)
            continue

        data = BagDataAnalyzer(bag_reader.bag_timestamps, bag_reader.extracted_data)

        total_dataset_list = data.sort_topic(NEED_TOPIC_BAG)

        for total_dataset in total_dataset_list:
        
            for key, value in total_dataset.items():
                if key == "steer":
                    print("this data size = " + str(len(value)))

            idx = 0

            dataset = CarDataset()

            for i in range(math.floor((len(total_dataset["xpos_x"]) // num_size))):

                # add need tag
                actual_size = min(num_size, len(total_dataset["xpos_x"]) - idx * num_size)
                dataset.data_logs["xvel_y"] = [0] * actual_size
                for key, value in total_dataset.items():
                    dataset.data_logs[key] = total_dataset[key][idx*num_size:idx*num_size+actual_size]

                # check data is cover
                # for key, value in dataset.data_logs.items():
                #     if np.isnan(value).any():
                #         print("key = " + str(key))
                #         continue

                # save data into file
                dataset.data_logs["lap_end"] = [0] * actual_size
                dataset.data_logs["lap_end"][-1] = 1
                now = datetime.datetime.now().isoformat(timespec='milliseconds')
                file_name_save = "bag_data_"  +str(now) + "_" + str(i) + ".pkl"
                file_savepath = os.path.join(file_save_path, file_name_save)

                for key, value in dataset.data_logs.items():
                    dataset.data_logs[key] = np.array(value)

                with open(file_savepath, 'wb') as outp: 
                    pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

                print("Saved Data to:", file_savepath)
                dataset.reset_logs()

                idx = idx + 1

            # if args.to_csv:
            #     bag_reader.to_csv(args.output)

    clear_unfit_pkl_file(file_save_path)
    print("transform bag to pkl data finish")