
# Generated: 2025-08-27T17:36:55.134734
# Source Brief: brief_01588.md
# Brief Index: 1588

        
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
        "Controls: Use ←→ arrows to select a column. Press Space to slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding bombs in this fast-paced arcade game. The more you slice, the harder it gets!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (15, 25, 35)
        self.COLOR_GRID = (30, 45, 60)
        self.COLOR_CURSOR = (255, 255, 255, 50)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.FRUIT_COLORS = {
            'apple': (140, 220, 70),  # Green
            'banana': (255, 230, 80), # Yellow
            'orange': (255, 160, 50), # Orange
            'grape': (160, 90, 220),  # Purple
        }
        self.BOMB_COLORS = {
            'red': (255, 80, 80),
            'green': (80, 255, 80),
            'blue': (80, 80, 255),
        }

        # Game constants
        self.MAX_SCORE = 200
        self.MAX_STEPS = 1000
        
        # Initialize state variables
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.cursor_pos = self.GRID_WIDTH // 2
        self.bomb_spawn_prob = 0.1
        self.fruit_spawn_prob = 0.02
        self.prev_space_held = False

        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_animations = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Reset per-step reward trackers
        self._reset_reward_trackers()

        # Update game state
        self._handle_input(action)
        self._update_game_state()
        
        # Calculate reward based on this step's events
        reward = self._calculate_reward()
        
        # Update score, which might trigger a win
        if self.sliced_fruit_points > 0:
            self.score = min(self.MAX_SCORE, self.score + self.sliced_fruit_points)
            if self.score >= self.MAX_SCORE and not self.game_won:
                self.game_won = True
                self.game_won_this_step = True
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_reward_trackers(self):
        self.sliced_fruit_points = 0
        self.was_empty_slice = False
        self.sliced_near_bomb = False
        self.bomb_sliced_this_step = False
        self.game_won_this_step = False

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif movement == 4:  # Right
            self.cursor_pos = min(self.GRID_WIDTH - 1, self.cursor_pos + 1)
            
        if space_held and not self.prev_space_held:
            self._perform_slice(self.cursor_pos)
            # sfx: slice_swoosh.wav
        
        self.prev_space_held = space_held

    def _perform_slice(self, column):
        sliced_something = False
        
        # Check for bombs first
        for bomb in self.bombs[:]:
            if bomb['col'] == column:
                self.bomb_sliced_this_step = True
                self._create_explosion(bomb['pos'], bomb['color'])
                self.bombs.remove(bomb)
                sliced_something = True
                # sfx: bomb_explosion.wav
        
        if self.bomb_sliced_this_step:
            return

        # Check for fruits
        sliced_fruits = [f for f in self.fruits if f['col'] == column]
        if sliced_fruits:
            sliced_something = True
            is_near_bomb = self._check_proximity_to_bombs(column)
            if is_near_bomb:
                self.sliced_near_bomb = True
                # sfx: bonus_slice.wav
            
            for fruit in sliced_fruits:
                self.sliced_fruit_points += fruit['points']
                self._create_fruit_splash(fruit['pos'], fruit['color'])
                self.fruits.remove(fruit)
                # sfx: fruit_splat.wav
        
        self.was_empty_slice = not sliced_something
        self._create_slice_animation(column)

    def _check_proximity_to_bombs(self, column):
        for bomb in self.bombs:
            if abs(bomb['col'] - column) == 1:
                return True
        return False

    def _update_game_state(self):
        # Update and remove old particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # Update slice animations
        for anim in self.slice_animations[:]:
            anim['lifespan'] -= 1
            if anim['lifespan'] <= 0:
                self.slice_animations.remove(anim)

        # Move falling objects
        for obj_list in [self.fruits, self.bombs]:
            for obj in obj_list[:]:
                obj['pos'][1] += obj['speed']
                if obj['pos'][1] > self.SCREEN_HEIGHT + 20:
                    obj_list.remove(obj)

        # Spawn new objects
        for col in range(self.GRID_WIDTH):
            if self.np_random.random() < self.bomb_spawn_prob:
                if sum(1 for b in self.bombs if b['col'] == col) == 0: # One bomb per column max
                    self._spawn_bomb(col)
            elif self.np_random.random() < self.fruit_spawn_prob:
                self._spawn_fruit(col)
        
        # Increase difficulty
        self.bomb_spawn_prob = min(0.3, self.bomb_spawn_prob + 0.0001)

    def _spawn_fruit(self, col):
        fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
        points_map = {'apple': 1, 'orange': 2, 'banana': 3, 'grape': 5}
        self.fruits.append({
            'col': col,
            'pos': [col * self.CELL_WIDTH + self.CELL_WIDTH / 2, -20],
            'speed': self.np_random.uniform(1.5, 3.5),
            'type': fruit_type,
            'color': self.FRUIT_COLORS[fruit_type],
            'points': points_map[fruit_type],
            'angle': 0
        })

    def _spawn_bomb(self, col):
        bomb_type = self.np_random.choice(list(self.BOMB_COLORS.keys()))
        self.bombs.append({
            'col': col,
            'pos': [col * self.CELL_WIDTH + self.CELL_WIDTH / 2, -20],
            'speed': self.np_random.uniform(2.0, 4.0),
            'type': bomb_type,
            'color': self.BOMB_COLORS[bomb_type],
            'fuse_timer': self.np_random.uniform(0, 2*math.pi)
        })

    def _calculate_reward(self):
        if self.bomb_sliced_this_step:
            return -100.0
        if self.game_won_this_step:
            return 100.0
        
        reward = 0.0
        if self.sliced_fruit_points > 0:
            reward += self.sliced_fruit_points # +1 to +5 per fruit
            if self.sliced_near_bomb:
                reward += 5.0
            else:
                reward -= 2.0
        elif self.was_empty_slice:
            reward -= 0.1
            
        return reward

    def _check_termination(self):
        return (self.bomb_sliced_this_step or 
                self.score >= self.MAX_SCORE or 
                self.steps >= self.MAX_STEPS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_WIDTH):
            x = i * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        
        # Draw cursor highlight
        cursor_rect = pygame.Rect(self.cursor_pos * self.CELL_WIDTH, 0, self.CELL_WIDTH, self.SCREEN_HEIGHT)
        s = pygame.Surface((self.CELL_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (cursor_rect.x, cursor_rect.y))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Draw falling objects
        for fruit in self.fruits:
            self._draw_fruit(fruit)
        for bomb in self.bombs:
            self._draw_bomb(bomb)

        # Draw slice animations
        for anim in self.slice_animations:
            alpha = max(0, min(255, int(255 * (anim['lifespan'] / 10.0))))
            x = anim['col'] * self.CELL_WIDTH + self.CELL_WIDTH // 2
            s = pygame.Surface((self.CELL_WIDTH // 2, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (255, 255, 255, alpha), s.get_rect(), border_radius=5)
            self.screen.blit(s, (x - self.CELL_WIDTH // 4, 0))

    def _draw_fruit(self, fruit):
        x, y = int(fruit['pos'][0]), int(fruit['pos'][1])
        color = fruit['color']
        outline_color = (min(255, c+50) for c in color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 18, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, 18, tuple(outline_color))

    def _draw_bomb(self, bomb):
        x, y = int(bomb['pos'][0]), int(bomb['pos'][1])
        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, 20, (30, 30, 30))
        pygame.gfxdraw.aacircle(self.screen, x, y, 20, (50, 50, 50))
        # Fuse
        bomb['fuse_timer'] += 0.3
        fuse_x = x + 5 * math.cos(bomb['fuse_timer'])
        fuse_y = y - 20 + 5 * math.sin(bomb['fuse_timer']/2)
        pygame.draw.line(self.screen, (150,150,100), (x, y-18), (fuse_x, fuse_y), 3)
        # Spark
        spark_color = (255, 255, 100) if int(self.steps/3) % 2 == 0 else (255,150,0)
        pygame.gfxdraw.filled_circle(self.screen, int(fuse_x), int(fuse_y), 3, spark_color)

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_large.render(score_text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 30))
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

    def _create_explosion(self, pos, color):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 8),
                'color': self.np_random.choice([(255,100,50), (255,200,50), (50,50,50)]),
                'lifespan': self.np_random.integers(20, 40)
            })

    def _create_fruit_splash(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'lifespan': self.np_random.integers(15, 30)
            })

    def _create_slice_animation(self, col):
        self.slice_animations.append({
            'col': col,
            'lifespan': 8
        })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "bomb_prob": self.bomb_spawn_prob,
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Human input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0 # Unused, but for completeness
        
        action = [movement, space_held, shift_held]

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Maintain frame rate
        clock.tick(30)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()