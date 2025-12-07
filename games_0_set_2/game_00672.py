
# Generated: 2025-08-27T14:25:42.831649
# Source Brief: brief_00672.md
# Brief Index: 672

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from itertools import combinations
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to select an intersection. Spacebar to allow Vertical traffic (↑↓), Shift for Horizontal traffic (←→)."
    )

    game_description = (
        "Direct traffic flow in a top-down simulation to maximize vehicle throughput while avoiding collisions."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_ROAD = (80, 80, 90)
        self.COLOR_LANE_MARKER = (120, 120, 130)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_GREEN_LIGHT = (0, 255, 100)
        self.COLOR_RED_LIGHT = (255, 50, 50)
        self.COLOR_LIGHT_OFF = (50, 50, 60)
        self.CAR_COLORS = [
            (50, 180, 255), (255, 180, 50), (200, 100, 255),
            (255, 255, 100), (100, 255, 200)
        ]

        # Game constants
        self.ROAD_WIDTH = 50
        self.CAR_SPEED = 2.5
        self.CAR_SIZE = (12, 22)
        self.MAX_EPISODE_STEPS = 60 * self.FPS # 60 seconds

        # Road layout
        self.H_ROADS_Y = [self.HEIGHT // 3, 2 * self.HEIGHT // 3]
        self.V_ROADS_X = [self.WIDTH // 3, 2 * self.WIDTH // 3]
        self.intersection_coords = sorted([(x, y) for x in self.V_ROADS_X for y in self.H_ROADS_Y])
        self.intersection_size = self.ROAD_WIDTH * 1.2
        
        # Initialize state variables that are not reset
        self.last_action = np.array([0, 0, 0])

        # Initialize the rest of the state
        self.reset()
        
        # self.validate_implementation() # Optional validation check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.collided_vehicles = []

        self.vehicles = []
        self.intersections = {
            pos: {'flow': 'vertical' if self.np_random.random() > 0.5 else 'horizontal'}
            for pos in self.intersection_coords
        }
        self.selected_intersection_idx = 0
        
        self.spawn_timer = 0.0
        self.spawn_rate = 0.5 # Start slower
        self.max_spawn_rate = 2.0
        self.next_vehicle_id = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            
            reward = self._calculate_reward()
            
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                if self.steps >= self.MAX_EPISODE_STEPS and not self.collided_vehicles:
                    self.win = True
                    reward += 100.0 # Win bonus
                elif self.collided_vehicles:
                    reward = -100.0 # Collision penalty
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect presses (transition from 0 to 1)
        movement_press = movement != 0 and self.last_action[0] == 0
        space_press = space_held and not (self.last_action[1] == 1)
        shift_press = shift_held and not (self.last_action[2] == 1)
        
        self.last_action = action

        if movement_press:
            num_intersections = len(self.intersection_coords)
            if movement in [1, 4]:  # Up or Right
                self.selected_intersection_idx = (self.selected_intersection_idx + 1) % num_intersections
            elif movement in [2, 3]:  # Down or Left
                self.selected_intersection_idx = (self.selected_intersection_idx - 1 + num_intersections) % num_intersections

        selected_pos = self.intersection_coords[self.selected_intersection_idx]
        if space_press:
            self.intersections[selected_pos]['flow'] = 'vertical'
            # sfx: light_switch_sfx()
        if shift_press:
            self.intersections[selected_pos]['flow'] = 'horizontal'
            # sfx: light_switch_sfx()

    def _update_game_state(self):
        self.steps += 1
        
        # Increase spawn rate over time
        self.spawn_rate = min(self.max_spawn_rate, self.spawn_rate + (0.001 / self.FPS) * self.FPS)

        # Spawn new vehicles
        self.spawn_timer += self.spawn_rate / self.FPS
        if self.spawn_timer >= 1.0:
            self.spawn_timer -= 1.0
            self._spawn_vehicle()

        # Update vehicles
        for v in self.vehicles:
            self._update_vehicle(v)
        
        # Remove despawned vehicles
        self.vehicles = [v for v in self.vehicles if v['alive']]

        # Check for collisions
        if not self.collided_vehicles:
            self._check_collisions()

    def _spawn_vehicle(self):
        road_choice = self.np_random.integers(4)
        is_vertical = road_choice < 2
        
        if is_vertical:
            road_x = self.np_random.choice(self.V_ROADS_X)
            start_y = -self.CAR_SIZE[1] if road_choice == 0 else self.HEIGHT + self.CAR_SIZE[1]
            start_pos = [road_x, start_y]
            
            dest_road_y = self.np_random.choice(self.H_ROADS_Y)
            dest_x = -self.CAR_SIZE[1] if self.np_random.random() < 0.5 else self.WIDTH + self.CAR_SIZE[1]
            end_pos = [dest_x, dest_road_y]
            path = [start_pos, [road_x, dest_road_y], end_pos]
            direction = 'vertical'
        else: # Horizontal
            road_y = self.np_random.choice(self.H_ROADS_Y)
            start_x = -self.CAR_SIZE[1] if road_choice == 2 else self.WIDTH + self.CAR_SIZE[1]
            start_pos = [start_x, road_y]
            
            dest_road_x = self.np_random.choice(self.V_ROADS_X)
            dest_y = -self.CAR_SIZE[1] if self.np_random.random() < 0.5 else self.HEIGHT + self.CAR_SIZE[1]
            end_pos = [dest_road_x, dest_y]
            path = [start_pos, [dest_road_x, road_y], end_pos]
            direction = 'horizontal'

        car = {
            'id': self.next_vehicle_id,
            'pos': np.array(start_pos, dtype=float),
            'size': self.CAR_SIZE if direction == 'vertical' else (self.CAR_SIZE[1], self.CAR_SIZE[0]),
            'color': self.np_random.choice(self.CAR_COLORS),
            'path': [np.array(p, dtype=float) for p in path],
            'path_index': 1,
            'speed': self.CAR_SPEED,
            'state': 'moving', # 'moving', 'stopped'
            'direction': direction,
            'alive': True,
            'crossed_intersections': set()
        }
        self.next_vehicle_id += 1
        self.vehicles.append(car)

    def _update_vehicle(self, v):
        if not v['alive']: return

        target_pos = v['path'][v['path_index']]
        direction_vec = target_pos - v['pos']
        distance = np.linalg.norm(direction_vec)

        # Intersection logic
        stop_dist = self.intersection_size / 2 + v['size'][1] * 0.75
        can_proceed = True
        for ix_pos_tuple in self.intersection_coords:
            if tuple(ix_pos_tuple) in v['crossed_intersections']: continue
            
            ix_pos = np.array(ix_pos_tuple)
            dist_to_ix = np.linalg.norm(v['pos'] - ix_pos)
            
            if dist_to_ix < stop_dist:
                intersection_state = self.intersections[ix_pos_tuple]
                if v['direction'] != intersection_state['flow']:
                    can_proceed = False
                    break
        
        v['state'] = 'stopped' if not can_proceed else 'moving'

        # Movement
        if v['state'] == 'moving':
            if distance < v['speed']:
                v['pos'] = target_pos
                if tuple(np.round(target_pos).astype(int)) in self.intersection_coords:
                    v['crossed_intersections'].add(tuple(np.round(target_pos).astype(int)))
                v['path_index'] += 1
                if v['path_index'] >= len(v['path']):
                    v['alive'] = False
            else:
                move_vec = (direction_vec / distance) * v['speed']
                v['pos'] += move_vec
        
        # Despawn if out of bounds for too long (failsafe)
        if not (-100 < v['pos'][0] < self.WIDTH + 100 and -100 < v['pos'][1] < self.HEIGHT + 100):
            v['alive'] = False

    def _check_collisions(self):
        if len(self.vehicles) < 2: return
        
        for v1, v2 in combinations(self.vehicles, 2):
            rect1 = pygame.Rect(v1['pos'][0] - v1['size'][0]/2, v1['pos'][1] - v1['size'][1]/2, v1['size'][0], v1['size'][1])
            rect2 = pygame.Rect(v2['pos'][0] - v2['size'][0]/2, v2['pos'][1] - v2['size'][1]/2, v2['size'][0], v2['size'][1])
            if rect1.colliderect(rect2):
                self.collided_vehicles = [v1['id'], v2['id']]
                # sfx: collision_sfx()
                return

    def _calculate_reward(self):
        reward = 0.0
        # Penalty for stopped cars
        stopped_cars = sum(1 for v in self.vehicles if v['state'] == 'stopped')
        reward -= stopped_cars * 0.1

        # Reward for passing through an intersection
        for v in self.vehicles:
            for ix_pos_tuple in v['crossed_intersections']:
                # This reward is tricky to grant only once.
                # A better approach is to reward when a car despawns successfully.
                pass
        
        # Reward for each car that successfully crossed an intersection and is now gone
        # This requires tracking cars from the previous step.
        # A simpler proxy: reward cars for moving inside an intersection zone.
        for v in self.vehicles:
            if v['state'] == 'moving':
                for ix_pos_tuple in self.intersection_coords:
                    dist_to_ix = np.linalg.norm(v['pos'] - np.array(ix_pos_tuple))
                    if dist_to_ix < self.intersection_size / 2:
                        reward += 0.1 # Small reward per frame for productive movement
                        break
        return reward

    def _check_termination(self):
        if self.collided_vehicles:
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "vehicles": len(self.vehicles),
            "spawn_rate": self.spawn_rate
        }

    def _render_game(self):
        # Draw roads
        for y in self.H_ROADS_Y:
            pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, y - self.ROAD_WIDTH // 2, self.WIDTH, self.ROAD_WIDTH))
            for i in range(0, self.WIDTH, 40):
                 pygame.draw.line(self.screen, self.COLOR_LANE_MARKER, (i, y), (i + 20, y), 2)

        for x in self.V_ROADS_X:
            pygame.draw.rect(self.screen, self.COLOR_ROAD, (x - self.ROAD_WIDTH // 2, 0, self.ROAD_WIDTH, self.HEIGHT))
            for i in range(0, self.HEIGHT, 40):
                 pygame.draw.line(self.screen, self.COLOR_LANE_MARKER, (x, i), (x, i + 20), 2)
        
        # Draw intersections and lights
        for i, pos in enumerate(self.intersection_coords):
            ix_rect = pygame.Rect(pos[0] - self.intersection_size/2, pos[1] - self.intersection_size/2, self.intersection_size, self.intersection_size)
            pygame.draw.rect(self.screen, self.COLOR_ROAD, ix_rect)
            
            state = self.intersections[pos]
            v_color = self.COLOR_GREEN_LIGHT if state['flow'] == 'vertical' else self.COLOR_RED_LIGHT
            h_color = self.COLOR_GREEN_LIGHT if state['flow'] == 'horizontal' else self.COLOR_RED_LIGHT
            
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] - self.ROAD_WIDTH*0.3), 5, v_color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] + self.ROAD_WIDTH*0.3), 5, v_color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0] - self.ROAD_WIDTH*0.3), int(pos[1]), 5, h_color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0] + self.ROAD_WIDTH*0.3), int(pos[1]), 5, h_color)

        # Draw selection highlight
        selected_pos = self.intersection_coords[self.selected_intersection_idx]
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        size = self.intersection_size * (1.1 + pulse * 0.2)
        alpha = 100 + pulse * 100
        sel_rect = pygame.Rect(0, 0, size, size)
        sel_rect.center = selected_pos
        
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 255, 0, alpha), s.get_rect(), border_radius=10)
        pygame.draw.rect(s, (255, 255, 0, 255), s.get_rect(), width=2, border_radius=10)
        self.screen.blit(s, sel_rect.topleft)

        # Draw vehicles
        for v in self.vehicles:
            color = v['color']
            if v['state'] == 'stopped':
                color = tuple(int(c * 0.6) for c in color)
            
            if v['id'] in self.collided_vehicles:
                color = self.COLOR_RED_LIGHT

            w, h = v['size']
            car_rect = pygame.Rect(v['pos'][0] - w/2, v['pos'][1] - h/2, w, h)
            pygame.draw.rect(self.screen, color, car_rect, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0), car_rect, width=1, border_radius=3)

    def _render_ui(self):
        time_left = (self.MAX_EPISODE_STEPS - self.steps) / self.FPS
        timer_text = self.font_small.render(f"Time Left: {max(0, time_left):.1f}s", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        selected_pos = self.intersection_coords[self.selected_intersection_idx]
        flow = self.intersections[selected_pos]['flow']
        hint_text = self.font_small.render(f"Selected: ({selected_pos[0]}, {selected_pos[1]}) | Flow: {flow.upper()}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hint_text, (self.WIDTH/2 - hint_text.get_width()/2, self.HEIGHT - 30))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "SUCCESS" if self.win else "COLLISION DETECTED"
            color = self.COLOR_GREEN_LIGHT if self.win else self.COLOR_RED_LIGHT
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Traffic Control")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    # Store the last frame's keyboard state to detect presses
    last_keys = pygame.key.get_pressed()

    while running:
        current_keys = pygame.key.get_pressed()
        
        movement = 0
        # Detect key presses for movement, not holds
        if current_keys[pygame.K_UP] and not last_keys[pygame.K_UP]: movement = 1
        elif current_keys[pygame.K_DOWN] and not last_keys[pygame.K_DOWN]: movement = 2
        elif current_keys[pygame.K_LEFT] and not last_keys[pygame.K_LEFT]: movement = 3
        elif current_keys[pygame.K_RIGHT] and not last_keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if current_keys[pygame.K_SPACE] else 0
        shift_held = 1 if current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        last_keys = current_keys

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and (terminated or truncated):
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        if terminated or truncated:
            if 'printed_end_message' not in locals() or not printed_end_message:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print("Press 'R' to restart.")
                printed_end_message = True
        else:
            printed_end_message = False

        clock.tick(env.FPS)
        
    env.close()