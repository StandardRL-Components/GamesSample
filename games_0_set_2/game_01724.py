
# Generated: 2025-08-27T18:04:35.271192
# Source Brief: brief_01724.md
# Brief Index: 1724

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "Bounce a ball off a paddle to break blocks in a fast-paced, isometric-2D arcade game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_BALLS = 3

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_OUTLINE = (180, 180, 180)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = {
            1: ((0, 220, 100), (0, 160, 70)),  # Green (fill, outline)
            3: ((0, 150, 255), (0, 100, 200)),  # Blue
            5: ((255, 80, 80), (200, 50, 50)),   # Red
        }

        # Game world dimensions (logical space)
        self.GAME_AREA_WIDTH = 300
        self.GAME_AREA_HEIGHT = 400
        self.PADDLE_Y = self.GAME_AREA_HEIGHT - 30
        self.PADDLE_WIDTH = 60
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.BALL_SPEED_INITIAL = 5

        # Isometric projection parameters
        self.ISO_TILE_W = 16
        self.ISO_TILE_H = 8
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 60

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # --- State variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        self.active_blocks_count = 0
        self.last_space_press = False

        self.reset()
        
        self.validate_implementation()

    def _to_iso(self, x, y):
        """Converts logical coordinates to isometric screen coordinates."""
        iso_x = self.ORIGIN_X + (x - y) * (self.ISO_TILE_W / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.ISO_TILE_H / 2)
        return int(iso_x), int(iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        
        # Paddle
        paddle_x = self.GAME_AREA_WIDTH / 2
        self.paddle = pygame.Rect(paddle_x - self.PADDLE_WIDTH / 2, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self.ball_launched = False
        self._reset_ball()

        # Blocks
        self.blocks = []
        block_w, block_h = 28, 12
        rows, cols = 6, 8
        
        points_map = {0: 5, 1: 5, 2: 3, 3: 3, 4: 1, 5: 1}

        for r in range(rows):
            for c in range(cols):
                points = points_map[r]
                color_fill, color_outline = self.BLOCK_COLORS[points]
                block_x = c * (block_w + 4) + (self.GAME_AREA_WIDTH - cols * (block_w + 4)) / 2 + 10
                block_y = r * (block_h + 4) + 50
                
                self.blocks.append({
                    "rect": pygame.Rect(block_x, block_y, block_w, block_h),
                    "points": points,
                    "color_fill": color_fill,
                    "color_outline": color_outline,
                    "active": True
                })
        self.active_blocks_count = len(self.blocks)
        
        # Particles
        self.particles = []
        self.last_space_press = False

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Attaches the ball to the center of the paddle."""
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_launched = False

    def step(self, action):
        reward = 0
        self.steps += 1
        
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Input ---
        if not self.game_over:
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.GAME_AREA_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

            # Launch ball on space press (rising edge)
            if space_held and not self.last_space_press and not self.ball_launched:
                self.ball_launched = True
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
                # sfx: launch_ball.wav
        
        self.last_space_press = space_held

        # --- Update Game State ---
        if not self.game_over:
            if self.ball_launched:
                self.ball_pos += self.ball_vel
                reward += self._handle_collisions()
            else:
                # Ball follows paddle before launch
                self.ball_pos.x = self.paddle.centerx

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Check Termination ---
        terminated = self.game_over
        if self.active_blocks_count == 0:
            reward += 100  # Win bonus
            terminated = True
            # sfx: win_game.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS < 0 or self.ball_pos.x + self.BALL_RADIUS > self.GAME_AREA_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.GAME_AREA_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: wall_bounce.wav
        if self.ball_pos.y - self.BALL_RADIUS < 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce.wav

        # Bottom wall (lose ball)
        if self.ball_pos.y + self.BALL_RADIUS > self.GAME_AREA_HEIGHT:
            self.balls_left -= 1
            reward -= 0.02 # Penalty for missing
            # sfx: lose_ball.wav
            if self.balls_left <= 0:
                self.game_over = True
                reward -= 100 # Lose penalty
                # sfx: game_over.wav
            else:
                self._reset_ball()
            return reward

        # Paddle collision
        paddle_rect_iso = self._get_iso_rect(self.paddle)
        if paddle_rect_iso.collidepoint(self._to_iso(self.ball_pos.x, self.ball_pos.y)) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            
            # Influence horizontal velocity
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.0
            
            # Normalize speed
            self.ball_vel.scale_to_length(self.BALL_SPEED_INITIAL)
            # sfx: paddle_hit.wav

        # Block collisions
        for block in self.blocks:
            if block['active']:
                block_rect_iso = self._get_iso_rect(block['rect'])
                if block_rect_iso.collidepoint(self._to_iso(self.ball_pos.x, self.ball_pos.y)):
                    block['active'] = False
                    self.active_blocks_count -= 1
                    
                    # Add score and reward
                    self.score += block['points']
                    reward += block['points'] + 0.1
                    
                    # Create particles
                    self._create_particles(block['rect'].center, block['color_fill'])
                    # sfx: block_break.wav
                    
                    # Bounce logic
                    # Determine if collision was more horizontal or vertical
                    ball_iso = self._to_iso(self.ball_pos.x, self.ball_pos.y)
                    dx = abs(ball_iso[0] - block_rect_iso.centerx)
                    dy = abs(ball_iso[1] - block_rect_iso.centery)
                    
                    if (dx / block_rect_iso.width) > (dy / block_rect_iso.height):
                        self.ball_vel.x *= -1
                    else:
                        self.ball_vel.y *= -1
                    
                    # Ensure ball is out of block
                    self.ball_pos += self.ball_vel * 1.1 
                    break # Only break one block per frame
        return reward
        
    def _create_particles(self, pos, color):
        iso_pos = self._to_iso(pos[0], pos[1])
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(iso_pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for i in range(0, self.GAME_AREA_WIDTH + self.GAME_AREA_HEIGHT, 20):
            # Lines going "down-right"
            p1 = self._to_iso(i, 0)
            p2 = self._to_iso(0, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            # Lines going "down-left"
            p1 = self._to_iso(self.GAME_AREA_WIDTH - i, 0)
            p2 = self._to_iso(self.GAME_AREA_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

    def _get_iso_rect(self, rect):
        """Get the vertices for an isometric polygon from a logical rect."""
        p1 = self._to_iso(rect.left, rect.top)
        p2 = self._to_iso(rect.right, rect.top)
        p3 = self._to_iso(rect.right, rect.bottom)
        p4 = self._to_iso(rect.left, rect.bottom)
        
        min_x = min(p1[0], p2[0], p3[0], p4[0])
        max_x = max(p1[0], p2[0], p3[0], p4[0])
        min_y = min(p1[1], p2[1], p3[1], p4[1])
        max_y = max(p1[1], p2[1], p3[1], p4[1])
        
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def _draw_iso_rect(self, rect, fill_color, outline_color):
        points = [
            self._to_iso(rect.left, rect.top),
            self._to_iso(rect.right, rect.top),
            self._to_iso(rect.right, rect.bottom),
            self._to_iso(rect.left, rect.bottom)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, fill_color)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            if block['active']:
                self._draw_iso_rect(block['rect'], block['color_fill'], block['color_outline'])

        # Paddle
        self._draw_iso_rect(self.paddle, self.COLOR_PADDLE, self.COLOR_PADDLE_OUTLINE)
        
        # Ball
        ball_iso_pos = self._to_iso(self.ball_pos.x, self.ball_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, ball_iso_pos[0], ball_iso_pos[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_iso_pos[0], ball_iso_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_iso_pos[0], ball_iso_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.box(self.screen, (int(p['pos'].x), int(p['pos'].y), p['size'], p['size']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            x = self.WIDTH - 25 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, x, 25, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, 25, 6, self.COLOR_BALL)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if self.active_blocks_count == 0:
                msg = "YOU WIN!"
                
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "active_blocks": self.active_blocks_count
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
        
        # Test game-specific logic
        self.reset()
        assert self.paddle.x >= 0 and self.paddle.right <= self.GAME_AREA_WIDTH
        self.step([3, 0, 0]) # Move left
        assert self.paddle.x < self.GAME_AREA_WIDTH / 2 - self.PADDLE_WIDTH / 2
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Game Started ---")
    print(env.user_guide)

    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Game ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"--- Episode Finished ---")
            print(f"Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # obs, info = env.reset() # Uncomment to auto-reset
            # total_reward = 0
            
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Control the human-playable framerate

    env.close()
    print("--- Game Closed ---")