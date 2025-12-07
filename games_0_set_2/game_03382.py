
# Generated: 2025-08-27T23:11:28.475360
# Source Brief: brief_03382.md
# Brief Index: 3382

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select the number under the cursor."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Number Ninja is a fast-paced arcade game where players must click procedurally generated numbers in ascending order within a time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_FPS = 60
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.TARGET_FPS
        self.NUM_TARGETS = 20
        self.CURSOR_SPEED = 6
        self.NUMBER_RADIUS = 25

        # --- Colors ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_ACTIVE_NUM = (255, 255, 255)
        self.COLOR_CLICKED_NUM = (80, 90, 100)
        self.COLOR_CURSOR = (0, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIMER_HIGH = (60, 220, 60)
        self.COLOR_TIMER_MID = (255, 200, 0)
        self.COLOR_TIMER_LOW = (220, 60, 60)
        self.COLOR_FLASH = (255, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_number = pygame.font.Font(None, 36)
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.time_remaining = None
        self.cursor_pos = None
        self.numbers = None
        self.next_target_number = None
        self.last_space_press = None
        self.screen_flash_timer = None
        self.particles = None
        
        # Initialize state variables
        self.reset()
        
        # --- Self-Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.next_target_number = 1
        self.last_space_press = False
        self.screen_flash_timer = 0
        self.particles = []
        self._generate_numbers()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            # If game is over, advance time for animations but don't change state
            self.steps += 1
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Time ---
        self.steps += 1
        self.time_remaining -= 1

        # --- Handle Input and Update State ---
        reward += self._handle_input(action)

        # --- Update Game Effects ---
        self._update_particles()
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1

        # --- Check for Termination Conditions ---
        terminated = False
        
        # 1. Win Condition
        if self.next_target_number > self.NUM_TARGETS:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        
        # 2. Timeout Condition
        elif self.time_remaining <= 0:
            self.game_over = True
            terminated = True
            reward = -50  # Timeout penalty
            
        # 3. Loss Condition (set during input handling)
        elif self.game_over:
            terminated = True
            reward = -100 # Incorrect click penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Movement & Distance Reward ---
        target = self._get_current_target()
        dist_before = float('inf')
        if target:
            dist_before = self.cursor_pos.distance_to(target['pos'])

        self._move_cursor(movement)

        if target:
            dist_after = self.cursor_pos.distance_to(target['pos'])
            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.1

        # --- Click Action ---
        is_click_event = space_held and not self.last_space_press
        self.last_space_press = space_held

        if is_click_event:
            # sfx_click
            clicked_num = self._get_number_under_cursor()
            if clicked_num:
                if clicked_num['value'] == self.next_target_number:
                    # Correct Click
                    # sfx_correct
                    reward += 1.0
                    self.score += 1
                    clicked_num['clicked'] = True
                    clicked_num['fade_timer'] = self.TARGET_FPS // 2  # 0.5 sec fade
                    self.next_target_number += 1
                    self._create_particles(clicked_num['pos'], self.COLOR_CURSOR)
                else:
                    # Incorrect Click
                    # sfx_error
                    self.game_over = True
                    self.screen_flash_timer = 15 # 0.25 sec flash
            else:
                # Clicked empty space
                # sfx_miss
                pass
        
        return reward

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: # Down
            self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: # Left
            self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: # Right
            self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _generate_numbers(self):
        self.numbers = []
        padding = self.NUMBER_RADIUS + 10
        min_dist = 2 * self.NUMBER_RADIUS

        for i in range(1, self.NUM_TARGETS + 1):
            attempts = 0
            while attempts < 200:
                pos = pygame.Vector2(
                    random.uniform(padding, self.WIDTH - padding),
                    random.uniform(padding + 40, self.HEIGHT - padding) # Avoid UI area
                )
                
                is_too_close = any(
                    pos.distance_to(n['pos']) < min_dist for n in self.numbers
                )

                if not is_too_close:
                    break
                attempts += 1
            
            self.numbers.append({
                'value': i,
                'pos': pos,
                'clicked': False,
                'fade_timer': 0
            })

    def _get_current_target(self):
        for num in self.numbers:
            if num['value'] == self.next_target_number:
                return num
        return None

    def _get_number_under_cursor(self):
        candidates = []
        for num in self.numbers:
            if not num['clicked'] and self.cursor_pos.distance_to(num['pos']) < self.NUMBER_RADIUS:
                candidates.append(num)
        
        if not candidates:
            return None
        
        # Return the one with the lowest value if overlapping
        return min(candidates, key=lambda x: x['value'])

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': velocity,
                'life': random.randint(15, 30), # 0.25 to 0.5 seconds
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Numbers ---
        for num in self.numbers:
            color = self.COLOR_ACTIVE_NUM
            if num['clicked']:
                color = self.COLOR_CLICKED_NUM
                if num['fade_timer'] > 0:
                    alpha = int(255 * (num['fade_timer'] / (self.TARGET_FPS / 2)))
                    color = (
                        max(0, min(255, self.COLOR_CLICKED_NUM[0] + alpha)),
                        max(0, min(255, self.COLOR_CLICKED_NUM[1] + alpha)),
                        max(0, min(255, self.COLOR_CLICKED_NUM[2] + alpha)),
                    )
                    num['fade_timer'] -= 1

            num_surf = self.font_number.render(str(num['value']), True, color)
            num_rect = num_surf.get_rect(center=num['pos'])
            self.screen.blit(num_surf, num_rect)

        # --- Draw Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['size'], p['size']))

        # --- Draw Cursor ---
        pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 12, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 12, self.COLOR_CURSOR + (50,))
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos_int[0] - 5, pos_int[1]), (pos_int[0] + 5, pos_int[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos_int[0], pos_int[1] - 5), (pos_int[0], pos_int[1] + 5), 1)

    def _render_ui(self):
        # --- Draw Score ---
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # --- Draw Timer Bar ---
        timer_percent = self.time_remaining / self.MAX_STEPS
        bar_width = (self.WIDTH - 220) * timer_percent
        bar_x = 110
        bar_y = 15
        
        if timer_percent > 0.5:
            color = self.COLOR_TIMER_HIGH
        elif timer_percent > 0.2:
            color = self.COLOR_TIMER_MID
        else:
            color = self.COLOR_TIMER_LOW

        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x - 2, bar_y - 2, self.WIDTH - 220 + 4, 14))
        if bar_width > 0:
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, 10))
            
        # --- Draw Target Number ---
        target_text = self.font_ui.render(f"Next: {self.next_target_number}", True, self.COLOR_UI_TEXT)
        target_rect = target_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(target_text, target_rect)

        # --- Draw Screen Flash on Error ---
        if self.screen_flash_timer > 0:
            alpha = int(100 * (self.screen_flash_timer / 15))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_FLASH + (alpha,))
            self.screen.blit(flash_surface, (0, 0))

        # --- Draw Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_TIMER_HIGH
            else:
                msg = "GAME OVER"
                color = self.COLOR_TIMER_LOW
                
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "next_target": self.next_target_number,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Number Ninja")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    terminated = False

    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame Surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.TARGET_FPS)

    env.close()