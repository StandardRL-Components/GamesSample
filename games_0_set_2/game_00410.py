
# Generated: 2025-08-27T13:34:20.646142
# Source Brief: brief_00410.md
# Brief Index: 410

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selected robot. Press Shift to cycle between robots (Red -> Green -> Blue)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide three robots to their charging stations. You have a limited number of moves. Plan your path carefully to solve the puzzle!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 10
        self.MAX_MOVES = 10
        self.NUM_ROBOTS = 3
        
        # Visual constants
        self.TILE_W_ISO = 32
        self.TILE_H_ISO = 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_WALL = (80, 88, 105)
        self.COLOR_WALL_TOP = (100, 110, 130)
        self.COLOR_STATION = (60, 60, 60)
        self.COLOR_STATION_LIT = (75, 95, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.ROBOT_COLORS = [(231, 76, 60), (46, 204, 113), (52, 152, 219)] # Red, Green, Blue
        self.ROBOT_GLOW = [(255, 120, 100), (90, 255, 160), (100, 190, 255)]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.robots = []
        self.stations = []
        self.walls = []
        self.selected_robot_idx = 0
        self.prev_shift_state = False
        self.last_action_info = {}

        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.selected_robot_idx = 0
        self.prev_shift_state = False
        self.last_action_info = {}
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        # Generate a set of unique, valid coordinates
        all_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_coords)
        
        # Place stations
        self.stations = [{'pos': all_coords.pop()} for _ in range(self.NUM_ROBOTS)]
        
        # Place robots
        self.robots = []
        for i in range(self.NUM_ROBOTS):
            self.robots.append({
                'pos': all_coords.pop(),
                'color': self.ROBOT_COLORS[i],
                'glow': self.ROBOT_GLOW[i],
                'on_station': False
            })

        # Place walls
        num_walls = self.np_random.integers(5, 10)
        self.walls = [{'pos': all_coords.pop()} for _ in range(num_walls)]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Robot Selection Cycling (on Shift press)
        if shift_held and not self.prev_shift_state:
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
        self.prev_shift_state = shift_held

        # 2. Handle Movement
        moved = False
        if movement > 0: # 0 is no-op
            self.moves_left -= 1
            moved = True
            
            robot = self.robots[self.selected_robot_idx]
            current_pos = list(robot['pos'])
            target_pos = list(current_pos)

            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right
            
            # Collision checks
            tx, ty = target_pos
            is_valid_move = True
            if not (0 <= tx < self.GRID_W and 0 <= ty < self.GRID_H): # Bounds check
                is_valid_move = False
            if tuple(target_pos) in [tuple(w['pos']) for w in self.walls]: # Wall check
                is_valid_move = False
            if tuple(target_pos) in [tuple(r['pos']) for r in self.robots]: # Other robot check
                is_valid_move = False
            
            if is_valid_move:
                robot['pos'] = tuple(target_pos)
        
        # 3. Calculate Rewards based on new state
        if moved:
            reward -= 1 # Cost for making a move
        
        all_robots_charged = True
        station_positions = [tuple(s['pos']) for s in self.stations]
        for robot in self.robots:
            was_on_station = robot.get('on_station', False)
            is_on_station = tuple(robot['pos']) in station_positions
            
            if is_on_station and not was_on_station:
                reward += 10 # Reward for newly charging a robot
            
            robot['on_station'] = is_on_station
            if not is_on_station:
                all_robots_charged = False

        # 4. Check for Termination
        terminated = False
        if all_robots_charged:
            reward += 50 # Win bonus
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward -= 50 # Lose penalty
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_ISO
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_ISO
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid, walls, stations, then robots to ensure correct layering
        render_list = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                render_list.append({'type': 'grid', 'pos': (x, y)})

        for wall in self.walls:
            render_list.append({'type': 'wall', 'pos': wall['pos']})
        
        for station in self.stations:
            render_list.append({'type': 'station', 'pos': station['pos']})

        for i, robot in enumerate(self.robots):
            render_list.append({'type': 'robot', 'robot_idx': i, 'pos': robot['pos']})
        
        # Sort by isometric depth (y-then-x) for correct occlusion
        render_list.sort(key=lambda item: (item['pos'][0] + item['pos'][1], item['pos'][1]))

        for item in render_list:
            x, y = item['pos']
            sx, sy = self._iso_to_screen(x, y)
            
            # Define isometric tile points
            points = [
                (sx, sy),
                (sx + self.TILE_W_ISO, sy + self.TILE_H_ISO),
                (sx, sy + 2 * self.TILE_H_ISO),
                (sx - self.TILE_W_ISO, sy + self.TILE_H_ISO)
            ]

            if item['type'] == 'grid':
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)
            
            elif item['type'] == 'station':
                is_occupied = item['pos'] in [tuple(r['pos']) for r in self.robots]
                color = self.COLOR_STATION_LIT if is_occupied else self.COLOR_STATION
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

            elif item['type'] == 'wall':
                # Draw 3D-like cube for walls
                wall_height = 40
                top_points = [(p[0], p[1] - wall_height) for p in points]
                pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_WALL_TOP)
                pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_GRID)
                # Draw front faces
                pygame.gfxdraw.filled_polygon(self.screen, [points[1], top_points[1], top_points[2], points[2]], self.COLOR_WALL)
                pygame.gfxdraw.filled_polygon(self.screen, [points[2], top_points[2], top_points[3], points[3]], self.COLOR_WALL)
                pygame.gfxdraw.aapolygon(self.screen, [points[1], top_points[1], top_points[2], points[2]], self.COLOR_GRID)
                pygame.gfxdraw.aapolygon(self.screen, [points[2], top_points[2], top_points[3], points[3]], self.COLOR_GRID)

            elif item['type'] == 'robot':
                robot = self.robots[item['robot_idx']]
                robot_center_y = sy + self.TILE_H_ISO - 15 # Adjust to appear centered on tile
                
                # Selection highlight
                if item['robot_idx'] == self.selected_robot_idx:
                    pulse = abs(math.sin(self.steps * 0.2))
                    radius = int(14 + 3 * pulse)
                    # Use gfxdraw for anti-aliased alpha blending
                    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*robot['glow'], 50 + int(pulse * 50)))
                    pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*robot['glow'], 100 + int(pulse * 100)))
                    self.screen.blit(temp_surf, (sx - radius, robot_center_y - radius))

                # Charging effect
                if robot['on_station']:
                    pygame.gfxdraw.aacircle(self.screen, sx, robot_center_y, 16, robot['glow'])
                    pygame.gfxdraw.aacircle(self.screen, sx, robot_center_y, 18, (*robot['glow'], 100))
                
                # Robot body
                pygame.gfxdraw.filled_circle(self.screen, sx, robot_center_y, 12, robot['color'])
                pygame.gfxdraw.aacircle(self.screen, sx, robot_center_y, 12, (0,0,0, 50))
                
                # Eye glint for liveliness
                pygame.gfxdraw.filled_circle(self.screen, sx + 3, robot_center_y - 3, 3, (255, 255, 255))

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (15, 15))
        
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)
        
        # Selected Robot Indicator
        for i in range(self.NUM_ROBOTS):
            color = self.ROBOT_COLORS[i]
            is_selected = (i == self.selected_robot_idx)
            radius = 12 if is_selected else 8
            alpha = 255 if is_selected else 150
            
            x_pos = 20 + i * 35
            y_pos = 60
            
            pygame.gfxdraw.filled_circle(self.screen, x_pos, y_pos, radius, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, x_pos, y_pos, radius, (*self.ROBOT_GLOW[i], alpha))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            win = all(r['on_station'] for r in self.robots)
            message = "PUZZLE SOLVED!" if win else "OUT OF MOVES"
            color = self.ROBOT_GLOW[1] if win else self.ROBOT_GLOW[0]
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            overlay.blit(end_text, end_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "robots_charged": sum(1 for r in self.robots if r['on_station'])
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game window
    render_mode = "human" # "rgb_array" or "human"
    
    # For human mode, we need a different screen setup
    if render_mode == "human":
        pygame.display.set_caption("Isometric Robot Puzzle")
        human_screen = pygame.display.set_mode((640, 400))

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    running = True
    while running:
        action = env.action_space.sample() # Default to random action
        
        # Human controls
        if render_mode == "human":
            keys = pygame.key.get_pressed()
            move_action = 0 # No-op
            if keys[pygame.K_UP]: move_action = 1
            elif keys[pygame.K_DOWN]: move_action = 2
            elif keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]
            
            # Only step on a key press for turn-based feel
            event_happened = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                        event_happened = True
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        print("--- Game Reset ---")
            
            if not event_happened and running:
                # If no event, just draw the screen and continue
                frame = np.transpose(env._get_observation(), (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                human_screen.blit(surf, (0, 0))
                pygame.display.flip()
                env.clock.tick(30)
                continue # Skip the step call if no relevant key was pressed
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render_mode == "human":
            # Render to the human-visible screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_left']}")

            if terminated:
                print("--- Episode Finished ---")
                pygame.time.wait(2000) # Pause on termination
                obs, info = env.reset()
                print("--- New Game Started ---")
            
            env.clock.tick(30) # Limit FPS
        else: # rgb_array mode for training
            if terminated:
                obs, info = env.reset()

    env.close()