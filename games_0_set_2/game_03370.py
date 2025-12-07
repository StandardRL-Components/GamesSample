import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce balls strategically to break all the blocks in a fast-paced isometric 2D arcade game."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_PADDLE = (255, 255, 0)
    COLOR_PADDLE_TOP = (255, 255, 120)
    COLOR_BALL = (255, 255, 255)
    COLOR_GRID = (40, 30, 60)
    BLOCK_COLORS = {
        1: {'main': (0, 255, 128), 'dark': (0, 180, 90)},   # Green
        2: {'main': (0, 128, 255), 'dark': (0, 90, 180)},   # Blue
        3: {'main': (255, 50, 100), 'dark': (180, 35, 70)}, # Red
    }
    COLOR_TEXT = (220, 220, 220)
    
    # Screen and World Dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_DEPTH = 18, 24
    ISO_TILE_WIDTH_HALF, ISO_TILE_HEIGHT_HALF = 16, 8
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 80
    BLOCK_HEIGHT = 16

    # Game Mechanics
    PADDLE_WIDTH = 4
    PADDLE_DEPTH = 1
    PADDLE_Y = WORLD_DEPTH - 2
    PADDLE_SPEED = 0.6
    BALL_SPEED = 0.4
    MAX_STEPS = 2000
    INITIAL_BALLS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle_x = 0.0
        self.balls_left = 0
        self.ball = None
        self.blocks = []
        self.particles = []
        self.space_was_held = False
        
        # self.reset() is called by the test harness, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.space_was_held = False
        
        self.paddle_x = (self.WORLD_WIDTH - self.PADDLE_WIDTH) / 2
        
        self._reset_ball()
        self._create_block_layout()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.02  # Time penalty to encourage faster play
        
        # 1. Handle Input
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_x += self.PADDLE_SPEED
        
        self.paddle_x = np.clip(self.paddle_x, 0, self.WORLD_WIDTH - self.PADDLE_WIDTH)

        # 2. Launch Ball
        if self.ball['on_paddle'] and space_held and not self.space_was_held:
            self.ball['on_paddle'] = False
            # Sound: Ball Launch
            self.ball['vy'] = -self.BALL_SPEED
            self.ball['vx'] = (self.np_random.random() - 0.5) * 0.1
        self.space_was_held = space_held

        # 3. Update Game Logic
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        # 4. Check Termination
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated and self.balls_left == 0 and not any(b['active'] for b in self.blocks):
             reward += 50 # Bonus for clearing screen on last ball
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_ball(self):
        self.ball = {
            'x': self.paddle_x + self.PADDLE_WIDTH / 2,
            'y': self.PADDLE_Y - 0.5,
            'vx': 0,
            'vy': 0,
            'on_paddle': True,
            'trail': deque(maxlen=10)
        }

    def _update_ball(self):
        if self.ball['on_paddle']:
            self.ball['x'] = self.paddle_x + self.PADDLE_WIDTH / 2
            self.ball['y'] = self.PADDLE_Y - 0.5
        else:
            self.ball['trail'].append(self._iso_to_screen(self.ball['x'], self.ball['y']))
            self.ball['x'] += self.ball['vx']
            self.ball['y'] += self.ball['vy']
            
    def _handle_collisions(self):
        if self.ball['on_paddle']:
            return 0
        
        reward = 0
        
        # Wall collisions
        if self.ball['x'] < 0 or self.ball['x'] > self.WORLD_WIDTH:
            self.ball['vx'] *= -1
            self.ball['x'] = np.clip(self.ball['x'], 0, self.WORLD_WIDTH)
            # Sound: Wall Bounce
        
        if self.ball['y'] < 0:
            self.ball['vy'] *= -1
            self.ball['y'] = np.clip(self.ball['y'], 0, self.WORLD_DEPTH)
            # Sound: Wall Bounce
            
        # Paddle collision
        if (self.paddle_x <= self.ball['x'] <= self.paddle_x + self.PADDLE_WIDTH and
                self.PADDLE_Y <= self.ball['y'] <= self.PADDLE_Y + self.PADDLE_DEPTH and
                self.ball['vy'] > 0):
            self.ball['vy'] *= -1
            
            # Add "spin" based on hit location
            hit_pos = (self.ball['x'] - self.paddle_x) / self.PADDLE_WIDTH
            self.ball['vx'] += (hit_pos - 0.5) * self.BALL_SPEED
            self.ball['vx'] = np.clip(self.ball['vx'], -self.BALL_SPEED*1.5, self.BALL_SPEED*1.5)
            # Sound: Paddle Hit
            
        # Block collisions
        prev_ball_x = self.ball['x'] - self.ball['vx']
        prev_ball_y = self.ball['y'] - self.ball['vy']
        
        for block in self.blocks:
            if not block['active']:
                continue
            
            if (block['x'] < self.ball['x'] < block['x'] + 1 and
                    block['y'] < self.ball['y'] < block['y'] + 1):
                
                block['active'] = False
                reward += block['points']
                self.score += block['points']
                # Sound: Block Break
                self._spawn_particles(block['x'] + 0.5, block['y'] + 0.5, self.BLOCK_COLORS[block['points']]['main'])
                
                # Determine which side was hit for bounce direction
                dx = (self.ball['x'] - (block['x'] + 0.5))
                dy = (self.ball['y'] - (block['y'] + 0.5))
                
                if abs(dx) > abs(dy): # Horizontal collision
                    self.ball['vx'] *= -1
                else: # Vertical collision
                    self.ball['vy'] *= -1
                
                break # Only handle one block collision per frame

        # Check for win
        if not any(b['active'] for b in self.blocks):
            reward += 50
            self.game_over = True
        
        # Check for loss
        if self.ball['y'] > self.WORLD_DEPTH:
            self.balls_left -= 1
            reward -= 5
            # Sound: Lose Ball
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
                
        return reward

    def _create_block_layout(self):
        self.blocks = []
        layout = [
            " R R R R R R R R ",
            " R B B B B B B R ",
            " R B G G G G B R ",
            " R B G - - G B R ",
            " R B G - - G B R ",
            " R B G G G G B R ",
            " R B B B B B B R ",
            " R R R R R R R R ",
        ]
        
        start_x = (self.WORLD_WIDTH - len(layout[0])) / 2
        start_y = 2
        
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                points = 0
                if char == 'G': points = 1
                elif char == 'B': points = 2
                elif char == 'R': points = 3
                
                if points > 0:
                    self.blocks.append({
                        'x': start_x + c,
                        'y': start_y + r,
                        'points': points,
                        'active': True
                    })

    def _spawn_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 0.2 + 0.05
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

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
            "balls_left": self.balls_left,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, surface, color_main, color_dark, wx, wy, ww, wd, wh):
        points = [
            self._iso_to_screen(wx, wy),
            self._iso_to_screen(wx + ww, wy),
            self._iso_to_screen(wx + ww, wy + wd),
            self._iso_to_screen(wx, wy + wd)
        ]
        
        top_face = points
        left_face = [points[3], points[2], (points[2][0], points[2][1] + wh), (points[3][0], points[3][1] + wh)]
        right_face = [points[2], points[1], (points[1][0], points[1][1] + wh), (points[2][0], points[2][1] + wh)]
        
        pygame.gfxdraw.aapolygon(surface, left_face, color_dark)
        pygame.gfxdraw.filled_polygon(surface, left_face, color_dark)
        pygame.gfxdraw.aapolygon(surface, right_face, color_dark)
        pygame.gfxdraw.filled_polygon(surface, right_face, color_dark)
        pygame.gfxdraw.aapolygon(surface, top_face, color_main)
        pygame.gfxdraw.filled_polygon(surface, top_face, color_main)

    def _render_game(self):
        # Draw grid
        for i in range(self.WORLD_WIDTH + 1):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.WORLD_DEPTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for i in range(self.WORLD_DEPTH + 1):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.WORLD_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        # Draw blocks
        for block in self.blocks:
            if block['active']:
                colors = self.BLOCK_COLORS[block['points']]
                self._draw_iso_rect(self.screen, colors['main'], colors['dark'], block['x'], block['y'], 1, 1, self.BLOCK_HEIGHT)

        # Draw paddle
        self._draw_iso_rect(self.screen, self.COLOR_PADDLE_TOP, self.COLOR_PADDLE, self.paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_DEPTH, self.BLOCK_HEIGHT // 2)

        # Draw ball trail
        if len(self.ball['trail']) > 1:
            for i in range(len(self.ball['trail']) - 1):
                alpha = int(255 * (i / len(self.ball['trail'])))
                color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], alpha)
                start_pos = self.ball['trail'][i]
                end_pos = self.ball['trail'][i+1]
                
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.aaline(temp_surf, color, start_pos, end_pos)
                self.screen.blit(temp_surf, (0,0))

        # Draw ball
        ball_pos = self._iso_to_screen(self.ball['x'], self.ball['y'])
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], 8, (100, 100, 255, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], 6, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos[0], ball_pos[1], 6, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            pos = self._iso_to_screen(p['x'], p['y'])
            alpha = max(0, int(255 * (p['life'] / 30)))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            
            temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, pos, 2)
            self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Balls left
        ball_icon_pos = self._iso_to_screen(0, 0)
        for i in range(self.balls_left):
            pos_x = self.SCREEN_WIDTH - 30 - (i * 20)
            pos_y = 25
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 6, self.COLOR_BALL)
            
        if self.game_over:
            win_text = "LEVEL CLEAR" if any(b['active'] for b in self.blocks) else "GAME OVER"
            if not any(b['active'] for b in self.blocks):
                win_text = "LEVEL CLEAR!"
            else:
                win_text = "GAME OVER"
            
            text_surf = self.font_large.render(win_text, True, self.COLOR_PADDLE_TOP)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    # This requires a display. Unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Action defaults
        movement = 0  # no-op
        space = 0     # released
        shift = 0     # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        if terminated:
            # Simple reset on key press after game over
            if any(keys):
                obs, info = env.reset()
                terminated = False
        else:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS

    env.close()