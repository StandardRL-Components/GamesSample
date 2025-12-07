
# Generated: 2025-08-27T21:01:25.775602
# Source Brief: brief_02652.md
# Brief Index: 2652

        
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
        "Controls: Use arrow keys to move. Push all the brown boxes onto the green targets before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down puzzle game. Race against the clock to solve the level by pushing every box to its target location."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.TILE_SIZE = 40
        self.FPS = 50
        self.MAX_STEPS = 1500  # 30 seconds * 50 FPS
        self.TIME_LIMIT = 30.0

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_BOX = (160, 100, 40)
        self.COLOR_TARGET_ACTIVE = (80, 200, 120)
        self.COLOR_TARGET_INACTIVE = (40, 100, 60)
        self.COLOR_PARTICLE = (255, 220, 50)
        self.COLOR_UI_TEXT = (230, 230, 230)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Level Layout
        self.walls = self._define_level_layout()
        self.initial_player_pos = (2, 5)
        self.initial_box_positions = [(4, 2), (7, 6), (12, 4)]
        self.target_positions = [(13, 2), (13, 5), (13, 8)]

        # State variables (will be initialized in reset)
        self.player_pos = None
        self.player_visual_pos = None
        self.boxes_pos = None
        self.boxes_visual_pos = None
        self.boxes_on_target = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0.0
        
        # Initialize state variables
        self.reset()
        
        # Validate after initialization
        self.validate_implementation()
    
    def _define_level_layout(self):
        walls = set()
        for x in range(self.GRID_WIDTH):
            walls.add((x, 0))
            walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(1, self.GRID_HEIGHT - 1):
            walls.add((0, y))
            walls.add((self.GRID_WIDTH - 1, y))
        
        # Internal walls
        for y in range(1, 4): walls.add((5, y))
        for y in range(6, 9): walls.add((10, y))
        return list(walls)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT

        self.player_pos = self.initial_player_pos
        self.player_visual_pos = self._grid_to_pixel(self.player_pos)

        self.boxes_pos = [pos for pos in self.initial_box_positions]
        self.boxes_visual_pos = [self._grid_to_pixel(pos) for pos in self.boxes_pos]
        self.boxes_on_target = [False] * len(self.boxes_pos)
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        
        if not self.game_over:
            self._handle_action(action)
            self._update_game_state()
            
            reward = self._calculate_reward()
            self.score += reward
            
            terminated = self._check_termination()
            if terminated:
                self.game_over = True
                # Apply terminal reward
                if all(self.boxes_on_target):
                    terminal_reward = 100.0
                else: # Time ran out
                    terminal_reward = -100.0
                reward += terminal_reward
                self.score += terminal_reward
        else:
            reward = 0.0
            terminated = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            self.game_over = True
            # Apply terminal reward if timed out on the very last step
            if not all(self.boxes_on_target):
                terminal_reward = -100.0
                reward += terminal_reward
                self.score += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        if not self._is_player_settled():
            return

        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        else: return

        next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        if next_pos in self.walls:
            return  # sound: bump_wall.wav

        if next_pos in self.boxes_pos:
            box_index = self.boxes_pos.index(next_pos)
            box_next_pos = (next_pos[0] + dx, next_pos[1] + dy)
            
            if box_next_pos in self.walls or box_next_pos in self.boxes_pos:
                return  # Can't push box
            
            # Push box
            self.boxes_pos[box_index] = box_next_pos
            self.player_pos = next_pos
            self._create_particles(self._grid_to_pixel(next_pos), 20) # sound: box_scrape.wav
        else:
            # Move player
            self.player_pos = next_pos # sound: footstep.wav

    def _update_game_state(self):
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)

        # Interpolate player
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        self.player_visual_pos = self._lerp_pos(self.player_visual_pos, target_pixel_pos, 0.4)

        # Interpolate boxes
        for i in range(len(self.boxes_pos)):
            target_pixel_pos = self._grid_to_pixel(self.boxes_pos[i])
            self.boxes_visual_pos[i] = self._lerp_pos(self.boxes_visual_pos[i], target_pixel_pos, 0.3)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        reward = -0.02  # Time penalty
        
        current_on_target_state = [False] * len(self.boxes_pos)
        for i, box_pos in enumerate(self.boxes_pos):
            if box_pos in self.target_positions:
                current_on_target_state[i] = True
                if not self.boxes_on_target[i]:
                    reward += 1.0  # New box on target
                    # sound: success_chime.wav
        self.boxes_on_target = current_on_target_state
        return reward

    def _check_termination(self):
        if all(self.boxes_on_target):
            return True
        if self.time_remaining <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw targets
        is_on_target_map = {pos: False for pos in self.target_positions}
        for box_pos in self.boxes_pos:
            if box_pos in is_on_target_map:
                is_on_target_map[box_pos] = True

        for pos in self.target_positions:
            pixel_pos = self._grid_to_pixel(pos)
            color = self.COLOR_TARGET_ACTIVE if is_on_target_map[pos] else self.COLOR_TARGET_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), self.TILE_SIZE // 2 - 4, color)
            pygame.gfxdraw.aacircle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), self.TILE_SIZE // 2 - 4, color)

        # Draw walls
        for pos in self.walls:
            rect = pygame.Rect(pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect, border_radius=2)
            
        # Draw particles
        for p in self.particles:
            size = max(1, int(p['life'] / 5))
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw boxes
        for pos in self.boxes_visual_pos:
            rect = pygame.Rect(0, 0, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
            rect.center = (int(pos[0]), int(pos[1]))
            pygame.draw.rect(self.screen, self.COLOR_BOX, rect, border_radius=4)
        
        # Draw player
        player_rect = pygame.Rect(0, 0, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
        player_rect.center = (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=6)
        
        glow_rect = player_rect.inflate(6, 6)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_PLAYER, 60), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)

    def _render_ui(self):
        # Time display
        time_text = f"TIME: {self.time_remaining:.2f}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 15, 10))

        # Box count display
        box_count = sum(self.boxes_on_target)
        box_text = f"TARGETS: {box_count} / {len(self.boxes_pos)}"
        box_surf = self.font_small.render(box_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(box_surf, (self.SCREEN_WIDTH - box_surf.get_width() - 15, 55))
        
        # Score display
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "LEVEL COMPLETE!" if all(self.boxes_on_target) else "TIME UP!"
            end_color = self.COLOR_TARGET_ACTIVE if all(self.boxes_on_target) else (255, 80, 80)
            end_surf = self.font_large.render(end_text, True, end_color)
            text_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "boxes_on_target": sum(self.boxes_on_target),
        }

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        return [x, y]

    def _lerp_pos(self, start, end, t):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return [start[0] + dx * t, start[1] + dy * t]
        
    def _is_player_settled(self):
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        dist_sq = (self.player_visual_pos[0] - target_pixel_pos[0])**2 + (self.player_visual_pos[1] - target_pixel_pos[1])**2
        return dist_sq < 2.0

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 21)
            })

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
        
        # print("✓ Implementation validated successfully")