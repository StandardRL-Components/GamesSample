
# Generated: 2025-08-28T02:50:59.548451
# Source Brief: brief_01831.md
# Brief Index: 1831

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to push the selected crystal. "
        "Space/Shift to cycle selection. Match the crystal patterns on the floor."
    )

    game_description = (
        "An isometric puzzle game. Push glowing crystals onto their matching targets "
        "before you run out of moves. Plan your moves carefully to solve each level."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        self.screen_width, self.screen_height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 50)
        self.font_game_over = pygame.font.Font(None, 80)

        # Visuals & Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_WALL_TOP = (60, 75, 105)
        self.COLOR_GRID = (25, 35, 55)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SELECT_GLOW = (255, 255, 255)
        self.CRYSTAL_PALETTE = {
            "red": ((255, 50, 50), (200, 20, 20), (150, 10, 10)),
            "blue": ((50, 150, 255), (20, 100, 200), (10, 75, 150)),
            "green": ((50, 255, 150), (20, 200, 100), (10, 150, 75)),
            "yellow": ((255, 255, 50), (200, 200, 20), (150, 150, 10)),
            "purple": ((200, 50, 255), (150, 20, 200), (100, 10, 150)),
            "cyan": ((50, 255, 255), (20, 200, 200), (10, 150, 150)),
        }

        # Isometric Grid
        self.grid_w, self.grid_h = 10, 10
        self.tile_w, self.tile_h = 60, 30
        self.origin_x = self.screen_width // 2
        self.origin_y = 80

        # Level Definitions
        self._define_levels()

        # State Variables (initialized in reset)
        self.level = 1
        self.score = 0
        self.steps = 0
        self.moves_left = 0
        self.game_over = False
        self.victory = False
        self.crystals = []
        self.targets = []
        self.selected_crystal_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def _define_levels(self):
        self.LEVELS = [
            None, # Level 1 is index 1
            {
                "moves": 10, "grid_size": (8, 8),
                "crystals": [((2, 5), "red"), ((5, 2), "blue")],
                "targets": [((2, 2), "red"), ((5, 5), "blue")]
            },
            {
                "moves": 12, "grid_size": (8, 8),
                "crystals": [((1, 1), "red"), ((1, 6), "blue"), ((6, 1), "green")],
                "targets": [((3, 3), "red"), ((3, 4), "blue"), ((4, 4), "green")]
            },
            {
                "moves": 15, "grid_size": (9, 9),
                "crystals": [((1, 4), "red"), ((4, 1), "blue"), ((7, 4), "green"), ((4, 7), "yellow")],
                "targets": [((3, 3), "red"), ((3, 5), "blue"), ((5, 3), "green"), ((5, 5), "yellow")]
            },
            {
                "moves": 18, "grid_size": (10, 10),
                "crystals": [((1, 1), "red"), ((1, 8), "blue"), ((8, 1), "green"), ((8, 8), "yellow"), ((4, 4), "purple")],
                "targets": [((3, 5), "red"), ((3, 3), "blue"), ((6, 5), "green"), ((6, 3), "yellow"), ((4, 8), "purple")]
            },
            {
                "moves": 20, "grid_size": (10, 10),
                "crystals": [((1, 2), "red"), ((2, 1), "blue"), ((1, 7), "green"), ((7, 1), "yellow"), ((8, 2), "purple"), ((2, 8), "cyan")],
                "targets": [((4, 4), "red"), ((5, 4), "blue"), ((4, 5), "green"), ((5, 5), "yellow"), ((4, 6), "purple"), ((5, 6), "cyan")]
            }
        ]

    def _load_level(self, level_num):
        if level_num > len(self.LEVELS) - 1:
            self.game_over = True
            self.victory = True
            return

        level_data = self.LEVELS[level_num]
        self.moves_left = level_data["moves"]
        self.grid_w, self.grid_h = level_data["grid_size"]
        
        # Sort by color to ensure crystal[i] matches target[i]
        sorted_crystals = sorted(level_data["crystals"], key=lambda x: x[1])
        sorted_targets = sorted(level_data["targets"], key=lambda x: x[1])

        self.crystals = [{"pos": pos, "color_key": color} for pos, color in sorted_crystals]
        self.targets = [{"pos": pos, "color_key": color} for pos, color in sorted_targets]
        
        self.selected_crystal_idx = 0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.level = 1
        self._load_level(self.level)
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        action_taken = False

        # Handle selection change (no move cost)
        num_crystals = len(self.crystals)
        if space_held and not self.prev_space_held:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % num_crystals
            action_taken = True
            # sound: "select_crystal.wav"
        elif shift_held and not self.prev_shift_held:
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + num_crystals) % num_crystals
            action_taken = True
            # sound: "select_crystal.wav"
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Handle movement (costs a move)
        if movement != 0:
            action_taken = True
            self.moves_left -= 1
            # sound: "push_start.wav"

            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # N, S, W, E
            dx, dy = dirs[movement]

            crystal_to_move = self.crystals[self.selected_crystal_idx]
            start_pos = crystal_to_move["pos"]
            
            # Calculate slide
            current_pos = start_pos
            while True:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                # Check bounds
                if not (0 <= next_pos[0] < self.grid_w and 0 <= next_pos[1] < self.grid_h):
                    break # Hit wall
                # Check other crystals
                if any(c["pos"] == next_pos for c in self.crystals):
                    break # Hit another crystal
                current_pos = next_pos
            
            end_pos = current_pos

            if start_pos != end_pos:
                # sound: "push_stop.wav"
                self._create_particles(start_pos, crystal_to_move["color_key"])
                
                # Calculate reward for this move
                reward += self._calculate_push_reward(self.selected_crystal_idx, start_pos, end_pos)
                crystal_to_move["pos"] = end_pos

            # Check for level completion
            if self._check_level_complete():
                reward += 100
                self.score += reward # Add final level reward before reset
                # sound: "level_complete.wav"
                self.level += 1
                if self.level > len(self.LEVELS) - 1:
                    self.game_over = True
                    self.victory = True
                    terminated = True
                else:
                    self._load_level(self.level)
                # Return immediately after level change to show new state
                return self._get_observation(), reward, terminated, False, self._get_info()

        # Check for termination conditions
        if self.moves_left <= 0 and not self._check_level_complete():
            self.game_over = True
            self.victory = False
            terminated = True
            reward -= 100 # Penalty for failing level
            # sound: "game_over.wav"
        
        if self.steps >= 5000: # Failsafe
            terminated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_push_reward(self, crystal_idx, start_pos, end_pos):
        target_pos = self.targets[crystal_idx]["pos"]
        
        dist_before = abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])
        dist_after = abs(end_pos[0] - target_pos[0]) + abs(end_pos[1] - target_pos[1])
        
        # +1 for getting closer, -1 for getting further
        reward = float(dist_before - dist_after)

        # +10 for landing on target
        if end_pos == target_pos:
            reward += 10.0
        # Check if it was on target before and moved off
        elif start_pos == target_pos and end_pos != target_pos:
            reward -= 10.0
            
        return reward

    def _check_level_complete(self):
        if not self.crystals:
            return False
        for i in range(len(self.crystals)):
            if self.crystals[i]["pos"] != self.targets[i]["pos"]:
                return False
        return True

    def _iso_to_screen(self, x, y):
        sx = self.origin_x + (x - y) * self.tile_w / 2
        sy = self.origin_y + (x + y) * self.tile_h / 2
        return int(sx), int(sy)

    def _draw_iso_cube(self, pos, color_key, size_mod=0):
        x, y = pos
        palette = self.CRYSTAL_PALETTE[color_key]
        face_color, side_color, top_color = palette
        
        tile_w = self.tile_w - size_mod
        tile_h = self.tile_h - size_mod
        
        center_x, center_y = self._iso_to_screen(x, y)
        
        # Points for the cube
        p = [
            (center_x, center_y - tile_h), # Top vertex
            (center_x + tile_w / 2, center_y - tile_h / 2),
            (center_x, center_y),
            (center_x - tile_w / 2, center_y - tile_h / 2),
            (center_x, center_y + tile_h), # Bottom vertex (hidden)
            (center_x - tile_w / 2, center_y + tile_h / 2),
            (center_x + tile_w / 2, center_y + tile_h / 2),
        ]
        
        # Draw top face
        pygame.gfxdraw.aapolygon(self.screen, (p[0], p[1], p[2], p[3]), top_color)
        pygame.gfxdraw.filled_polygon(self.screen, (p[0], p[1], p[2], p[3]), top_color)
        
        # Draw left face
        pygame.gfxdraw.aapolygon(self.screen, (p[3], p[2], p[5], (p[5][0], p[5][1]-tile_h)), side_color)
        pygame.gfxdraw.filled_polygon(self.screen, (p[3], p[2], p[5], (p[5][0], p[5][1]-tile_h)), side_color)

        # Draw right face
        pygame.gfxdraw.aapolygon(self.screen, (p[1], p[2], p[6], (p[6][0], p[6][1]-tile_h)), face_color)
        pygame.gfxdraw.filled_polygon(self.screen, (p[1], p[2], p[6], (p[6][0], p[6][1]-tile_h)), face_color)

    def _draw_target(self, pos, color_key):
        palette = self.CRYSTAL_PALETTE[color_key]
        color = palette[1] # Use a muted version
        center_x, center_y = self._iso_to_screen(pos[0], pos[1])
        points = [
            (center_x, center_y + self.tile_h / 2),
            (center_x + self.tile_w / 2, center_y),
            (center_x, center_y - self.tile_h / 2),
            (center_x - self.tile_w / 2, center_y),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        for i in range(4): # Thicken line
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]+(i%2), p[1]+(i//2)) for p in points], color)

    def _create_particles(self, pos, color_key):
        center_x, center_y = self._iso_to_screen(pos[0], pos[1])
        color = self.CRYSTAL_PALETTE[color_key][0]
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['life'] / 6))
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor and walls for context
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                center_x, center_y = self._iso_to_screen(x, y)
                points = [
                    (center_x, center_y + self.tile_h / 2),
                    (center_x + self.tile_w / 2, center_y),
                    (center_x, center_y - self.tile_h / 2),
                    (center_x - self.tile_w / 2, center_y),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Combine all drawable objects to sort them for correct rendering order
        draw_queue = []
        for target in self.targets:
            draw_queue.append(('target', target))
        for i, crystal in enumerate(self.crystals):
            draw_queue.append(('crystal', crystal, i == self.selected_crystal_idx))
        
        # Sort by grid y-coordinate, then x, to ensure correct isometric overlap
        def sort_key(item):
            obj_type, data = item[0], item[1]
            pos = data['pos']
            # Draw targets below crystals at the same spot
            z_order = 0 if obj_type == 'target' else 1
            return (pos[1], pos[0], z_order)

        draw_queue.sort(key=sort_key)

        for item in draw_queue:
            obj_type = item[0]
            if obj_type == 'target':
                _, target_data = item
                self._draw_target(target_data['pos'], target_data['color_key'])
            elif obj_type == 'crystal':
                _, crystal_data, is_selected = item
                self._draw_iso_cube(crystal_data['pos'], crystal_data['color_key'])
                if is_selected and not self.game_over:
                    center_x, center_y = self._iso_to_screen(crystal_data['pos'][0], crystal_data['pos'][1])
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y + int(self.tile_h/2) + 5, int(self.tile_w/2), self.COLOR_SELECT_GLOW)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y + int(self.tile_h/2) + 5, int(self.tile_w/2)-1, self.COLOR_SELECT_GLOW)
        
        self._update_and_draw_particles()

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        # Level
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.screen_width - level_text.get_width() - 20, 20))
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 55))
        
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.victory:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            game_over_surf = self.font_game_over.render(msg, True, color)
            self.screen.blit(game_over_surf, (self.screen_width/2 - game_over_surf.get_width()/2, self.screen_height/2 - game_over_surf.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    render_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Crystal Cavern")
    
    action = [0, 0, 0] # No-op, release all
    
    print("--- Crystal Cavern ---")
    print(env.game_description)
    print(env.user_guide)

    while not done:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Get key presses for this frame
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = [movement, space_held, shift_held]
        
        # Only step if an action is taken (for turn-based play)
        if current_action != [0,0,0]:
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Level: {info['level']}, Moves: {info['moves_left']}")
        
        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()