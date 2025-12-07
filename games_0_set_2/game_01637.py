
# Generated: 2025-08-28T02:15:28.893722
# Source Brief: brief_01637.md
# Brief Index: 1637

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select/place a ball. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Sort the colored balls into their matching slots before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 12
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 24, 12
        self.ORIGIN_X, self.ORIGIN_Y = self.SCREEN_WIDTH // 2, 80
        self.TOTAL_STAGES = 3
        self.MOVES_PER_STAGE = 50

        # Colors
        self.COLORS = [
            (255, 87, 87), (255, 170, 87), (255, 255, 87),
            (87, 255, 87), (87, 255, 255), (87, 170, 255),
            (87, 87, 255), (170, 87, 255), (255, 87, 255),
            (255, 128, 192)
        ]
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)

        # Initialize state variables
        self.stage = 0
        self.moves_left = 0
        self.cumulative_score = 0
        self.cursor_pos = [0, 0]
        self.balls = []
        self.slots = []
        self.selected_ball_idx = None
        self.selected_ball_original_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.game_over = False
        self.win = False
        self.steps = 0
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.cumulative_score = 0
        self.stage = 1
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.last_shift_held = False
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.moves_left = self.MOVES_PER_STAGE
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_ball_idx = None
        self.selected_ball_original_pos = None
        self.balls = []
        self.slots = []

        num_entities = len(self.COLORS) * 2
        possible_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        chosen_indices = self.np_random.choice(len(possible_coords), size=num_entities, replace=False)
        chosen_coords = [possible_coords[i] for i in chosen_indices]

        for i in range(len(self.COLORS)):
            self.slots.append({
                'pos': list(chosen_coords[i]),
                'color': self.COLORS[i],
                'filled': False
            })
            self.balls.append({
                'pos': list(chosen_coords[i + len(self.COLORS)]),
                'color': self.COLORS[i],
            })
    
    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if shift_press and self.selected_ball_idx is not None:
            # Deselect
            self.selected_ball_idx = None
            self.selected_ball_original_pos = None
            # sfx: deselect_sound

        elif space_press:
            if self.selected_ball_idx is None:
                # Try to select a ball
                for i, ball in enumerate(self.balls):
                    if ball['pos'] == self.cursor_pos:
                        self.selected_ball_idx = i
                        self.selected_ball_original_pos = list(ball['pos'])
                        # sfx: select_sound
                        break
            else:
                # Try to place the selected ball
                self.moves_left -= 1
                ball_to_place = self.balls[self.selected_ball_idx]
                target_pos = self.cursor_pos

                # Check for placement on a slot
                target_slot = next((s for s in self.slots if s['pos'] == target_pos), None)
                
                is_occupied_by_ball = any(b['pos'] == target_pos for i, b in enumerate(self.balls) if i != self.selected_ball_idx)

                if target_slot and not target_slot['filled'] and target_slot['color'] == ball_to_place['color']:
                    # 1. Correct placement
                    reward += 1.0
                    target_slot['filled'] = True
                    ball_to_place['pos'] = [-1, -1] # Remove ball from play
                    # sfx: success_sound
                elif (target_slot and (target_slot['filled'] or target_slot['color'] != ball_to_place['color'])) or is_occupied_by_ball:
                    # 2. Invalid move (occupied or wrong slot)
                    reward -= 0.1
                    # Ball does not move
                    # sfx: error_sound
                else:
                    # 3. Valid move to an empty space
                    ball_to_place['pos'] = list(target_pos)
                    # sfx: place_sound

                self.selected_ball_idx = None
                self.selected_ball_original_pos = None

        # --- Check Game State ---
        if all(s['filled'] for s in self.slots):
            reward += 10
            self.stage += 1
            if self.stage > self.TOTAL_STAGES:
                self.win = True
                terminated = True
                reward += 100
                # sfx: game_win_sound
            else:
                self._setup_stage()
                # sfx: stage_complete_sound

        if self.moves_left <= 0 and not terminated and not all(s['filled'] for s in self.slots):
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: game_over_sound

        self.cumulative_score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)
    
    def _get_rhombus_points(self, x, y):
        p1 = self._iso_to_screen(x, y)
        p2 = self._iso_to_screen(x + 1, y)
        p3 = self._iso_to_screen(x + 1, y + 1)
        p4 = self._iso_to_screen(x, y + 1)
        return [p1, p2, p3, p4]

    def _render_grid(self):
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_slots(self):
        for slot in self.slots:
            points = self._get_rhombus_points(slot['pos'][0], slot['pos'][1])
            color = slot['color']
            if slot['filled']:
                color = tuple(c // 2.5 for c in color)
            
            pygame.draw.polygon(self.screen, color, points)
            border_color = tuple(min(255, c + 30) for c in self.COLOR_GRID)
            pygame.draw.polygon(self.screen, border_color, points, 2)

    def _render_balls(self):
        for i, ball in enumerate(self.balls):
            if ball['pos'][0] < 0: continue

            pos_x, pos_y = ball['pos']
            is_selected = self.selected_ball_idx == i
            
            if is_selected:
                pos_x, pos_y = self.cursor_pos

            center_x, center_y = self._iso_to_screen(pos_x + 0.5, pos_y + 0.5)
            radius = self.TILE_HEIGHT_HALF + (2 if is_selected else 0)
            
            # Shadow
            shadow_y = center_y + 5 + (3 if is_selected else 0)
            shadow_radius_x = radius
            shadow_radius_y = radius // 2
            shadow_rect = pygame.Rect(0, 0, shadow_radius_x * 2, shadow_radius_y * 2)
            shadow_rect.center = (center_x, shadow_y)
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (0, 0, *shadow_rect.size))
            self.screen.blit(shadow_surf, shadow_rect.topleft)

            # Ball with highlight
            ball_center_y = center_y - (15 if is_selected else 0)
            pygame.gfxdraw.filled_circle(self.screen, center_x, ball_center_y, radius, ball['color'])
            pygame.gfxdraw.aacircle(self.screen, center_x, ball_center_y, radius, ball['color'])
            highlight_x = center_x - radius // 3
            highlight_y = ball_center_y - radius // 3
            pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, radius // 3, (255, 255, 255, 120))
            
    def _render_cursor(self):
        points = self._get_rhombus_points(self.cursor_pos[0], self.cursor_pos[1])
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 3)

        if self.selected_ball_idx is not None:
            ball_color = self.balls[self.selected_ball_idx]['color']
            center_x, _ = self._iso_to_screen(self.cursor_pos[0] + 0.5, self.cursor_pos[1] + 0.5)
            cursor_y = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])[1]
            indicator_y = cursor_y - 40
            pygame.gfxdraw.filled_circle(self.screen, center_x, indicator_y, 6, ball_color)
            pygame.gfxdraw.aacircle(self.screen, center_x, indicator_y, 6, self.COLOR_TEXT)

    def _render_ui(self):
        stage_text = self.font_medium.render(f"Stage: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        score_text = self.font_medium.render(f"Score: {self.cumulative_score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        moves_text = self.font_medium.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 40))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win:
            end_text_str, color = "YOU WIN!", self.COLOR_WIN
        else:
            end_text_str, color = "GAME OVER", self.COLOR_LOSE

        end_text = self.font_large.render(end_text_str, True, color)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(end_text, text_rect)

        score_text = self.font_medium.render(f"Final Score: {self.cumulative_score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
        self.screen.blit(score_text, score_rect)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_slots()
        self._render_balls()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over or self.win:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.cumulative_score,
            "steps": self.steps,
            "stage": self.stage,
            "moves_left": self.moves_left,
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

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Ball Sorter")
    clock = pygame.time.Clock()

    action = [0, 0, 0]  # No-op, no space, no shift

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            # --- Key State Handling ---
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space, shift]

            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
            if terminated:
                print("Episode finished!")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for manual play

    env.close()