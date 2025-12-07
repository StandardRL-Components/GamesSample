
# Generated: 2025-08-28T04:48:27.943088
# Source Brief: brief_05368.md
# Brief Index: 5368

        
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
        "Controls: Use arrow keys to move the monster. Eat all the food before the timer runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a hungry monster through a grid to devour all the food before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visual & Game Constants
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_MONSTER = (50, 205, 50)
        self.COLOR_MONSTER_EYE = (0, 0, 0)
        self.COLOR_FOOD = (255, 60, 60)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TIMER = (255, 220, 0)

        self.GRID_W, self.GRID_H = 20, 12
        self.CELL_SIZE = 30
        self.MARGIN_TOP = 50
        self.MARGIN_SIDES = (self.screen_width - self.GRID_W * self.CELL_SIZE) // 2

        self.FONT_UI = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_GAME_OVER = pygame.font.SysFont("Verdana", 50, bold=True)

        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.TIME_LIMIT_SECONDS
        self.NUM_FOOD = 25
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.monster_pos = [0, 0]
        self.food_items = []
        self.particles = []
        self.monster_anim = {'squash_timer': 0, 'last_move_dir': (0, 0)}
        
        # Initialize state variables
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        
        # Place monster in the center
        self.monster_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        # Generate food positions
        all_positions = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        all_positions.remove(tuple(self.monster_pos))
        self.np_random.shuffle(all_positions)
        self.food_items = [list(pos) for pos in all_positions[:self.NUM_FOOD]]

        self.monster_anim = {'squash_timer': 0, 'last_move_dir': (1, 0)} # Start facing right
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        self.steps += 1
        reward = 0.0
        terminated = False

        old_pos = self.monster_pos[:]
        old_dist_to_food = self._get_dist_to_nearest_food()

        # --- Update game logic ---
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            
            new_x = self.monster_pos[0] + dx
            new_y = self.monster_pos[1] + dy

            # Clamp to grid boundaries
            new_x = max(0, min(self.GRID_W - 1, new_x))
            new_y = max(0, min(self.GRID_H - 1, new_y))
            
            self.monster_pos = [new_x, new_y]

            # Trigger animation if moved
            if self.monster_pos != old_pos:
                self.monster_anim['squash_timer'] = 6 # 6 frames of animation
                self.monster_anim['last_move_dir'] = (dx, dy)

        # Distance-based reward (if monster moved)
        if self.monster_pos != old_pos:
            new_dist_to_food = self._get_dist_to_nearest_food()
            # Reward for getting closer, penalize for getting further
            reward += (old_dist_to_food - new_dist_to_food) * 0.1 

        # Food consumption
        if self.monster_pos in self.food_items:
            self.food_items.remove(self.monster_pos)
            self.score += 1
            reward += 10.0
            # sfx: munch.wav
            pixel_pos = self._grid_to_pixel(self.monster_pos, center=True)
            self._spawn_particles(pixel_pos, self.COLOR_FOOD)

        # Update animations
        self._update_monster_animation()
        self._update_particles()
        
        # --- Check termination conditions ---
        if not self.food_items: # Win condition
            terminated = True
            self.game_over = True
            reward += 50.0
        
        if self.steps >= self.MAX_STEPS: # Lose condition
            terminated = True
            self.game_over = True
            if self.food_items: # Only penalize if they didn't win on the last step
                reward -= 50.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "food_remaining": len(self.food_items),
        }

    def _update_monster_animation(self):
        if self.monster_anim['squash_timer'] > 0:
            self.monster_anim['squash_timer'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # pos_x += vel_x
            p[1] += p[3] # pos_y += vel_y
            p[4] -= 1    # life -= 1
            p[2] *= 0.95 # friction
            p[3] *= 0.95

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = self.MARGIN_SIDES + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.MARGIN_TOP), (px, self.MARGIN_TOP + self.GRID_H * self.CELL_SIZE))
        for y in range(self.GRID_H + 1):
            py = self.MARGIN_TOP + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.MARGIN_SIDES, py), (self.MARGIN_SIDES + self.GRID_W * self.CELL_SIZE, py))

        # Draw food
        for food_pos in self.food_items:
            px, py = self._grid_to_pixel(food_pos, center=True)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), self.CELL_SIZE // 3, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), self.CELL_SIZE // 3, self.COLOR_FOOD)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 20.0))))
            color = (*p[5], alpha)
            size = max(1, int(self.CELL_SIZE / 8 * (p[4] / 20.0)))
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

        # Draw monster
        monster_px, monster_py = self._grid_to_pixel(self.monster_pos, center=True)
        
        # Breathing animation
        breath = math.sin(self.steps * 0.15) * 0.05 + 1.0
        
        # Squash and stretch animation
        squash_factor = 0
        if self.monster_anim['squash_timer'] > 0:
            t = self.monster_anim['squash_timer'] / 6.0
            squash_factor = math.sin(t * math.pi) * 0.4

        dir_x, dir_y = self.monster_anim['last_move_dir']
        
        # If moving horizontally, squash vertically and stretch horizontally
        w = self.CELL_SIZE * 0.45 * breath * (1 + squash_factor * abs(dir_x))
        h = self.CELL_SIZE * 0.45 * breath * (1 - squash_factor * abs(dir_x))
        # If moving vertically, squash horizontally and stretch vertically
        w *= (1 - squash_factor * abs(dir_y))
        h *= (1 + squash_factor * abs(dir_y))

        monster_rect = pygame.Rect(monster_px - w, monster_py - h, w * 2, h * 2)
        pygame.draw.ellipse(self.screen, self.COLOR_MONSTER, monster_rect)

        # Draw eye
        eye_offset_x = dir_x * w * 0.4
        eye_offset_y = dir_y * h * 0.4
        eye_size = max(2, int(self.CELL_SIZE * 0.1))
        pygame.gfxdraw.filled_circle(self.screen, int(monster_px + eye_offset_x), int(monster_py + eye_offset_y), eye_size, self.COLOR_MONSTER_EYE)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score * 100}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        secs = int(time_left % 60)
        timer_text = self.FONT_UI.render(f"TIME: {secs:02d}", True, self.COLOR_TIMER)
        timer_rect = timer_text.get_rect(right=self.screen_width - 20, top=10)
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            if not self.food_items:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = (255, 100, 100)
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.FONT_GAME_OVER.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _grid_to_pixel(self, grid_pos, center=False):
        px = self.MARGIN_SIDES + grid_pos[0] * self.CELL_SIZE
        py = self.MARGIN_TOP + grid_pos[1] * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return px, py
    
    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append([pos[0], pos[1], vx, vy, life, color])

    def _get_dist_to_nearest_food(self):
        if not self.food_items:
            return 0
        
        monster_x, monster_y = self.monster_pos
        min_dist = float('inf')
        for food_x, food_y in self.food_items:
            dist = abs(monster_x - food_x) + abs(monster_y - food_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display window
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Monster Munch")
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']*100}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Clock ---
        env.clock.tick(env.FPS)
        
    pygame.quit()