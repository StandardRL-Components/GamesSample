import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
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
        "A fast-paced, neon-drenched block-breaking game. Clear stages by destroying all blocks, but watch the timer and don't lose your last ball!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 9000  # 60s * 3 stages * 50fps buffer

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (40, 0, 70)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (200, 200, 255, 50)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150, 100)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_WALL = (100, 100, 150)
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Pre-render background
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.WIDTH, y))

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = None
        self.ball_speed_base = None
        self.ball_launched = None
        self.blocks = []
        self.particles = []
        self.stage = 0
        self.balls_left = 0
        self.stage_timer = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.np_random = None # Will be seeded in reset
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # This is for debugging and not needed in the final version
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = 3
        
        self._setup_paddle()
        self._setup_stage()
        self._setup_new_life()
        
        return self._get_observation(), self._get_info()

    def _setup_paddle(self):
        paddle_w, paddle_h = 100, 15
        self.paddle = pygame.Rect((self.WIDTH - paddle_w) / 2, self.HEIGHT - 40, paddle_w, paddle_h)
        self.paddle_speed = 12

    def _setup_stage(self):
        self.blocks.clear()
        self.particles.clear()
        self.stage_timer = 60 * self.FPS

        num_blocks, block_hp = 0, 0
        if self.stage == 1:
            num_blocks, block_hp = 50, 1
        elif self.stage == 2:
            num_blocks, block_hp = 60, 2
        elif self.stage == 3:
            num_blocks, block_hp = 70, 3

        block_w, block_h = 50, 20
        gap = 4
        cols = 10
        rows = math.ceil(num_blocks / cols)
        grid_w = cols * (block_w + gap) - gap
        start_x = (self.WIDTH - grid_w) / 2
        start_y = 60

        for i in range(num_blocks):
            row = i // cols
            col = i % cols
            
            # Complex patterns for higher stages
            if self.stage == 2 and (row % 2 == 0) == (col % 2 == 0):
                continue
            if self.stage == 3 and (col < 2 or col > 7):
                continue

            hue = (i * 15) % 360
            color = pygame.Color(0)
            color.hsla = (hue, 100, 50, 100)
            
            block_rect = pygame.Rect(
                start_x + col * (block_w + gap),
                start_y + row * (block_h + gap),
                block_w, block_h
            )
            self.blocks.append({"rect": block_rect, "health": block_hp, "color": color, "max_health": block_hp})

    def _setup_new_life(self):
        self.ball_radius = 8
        self.ball_speed_base = 6
        self.ball_launched = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.ball_radius - 1]
        self.ball_vel = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, only care about press
        # shift_held is unused per brief

        reward = -0.02 # Time penalty
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += self.paddle_speed
        self.paddle.clamp_ip(self.screen.get_rect())

        if space_pressed and not self.ball_launched:
            # sfx: launch_ball.wav
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [self.ball_speed_base * math.cos(angle), self.ball_speed_base * math.sin(angle)]

        # 2. Update game logic
        self.steps += 1
        self.stage_timer -= 1
        self._update_particles()

        if self.ball_launched:
            reward += self._update_ball()
        else:
            self.ball_pos[0] = self.paddle.centerx

        # RL-specific reward shaping
        if self.ball_vel[1] > 0: # Ball is descending
            if abs(self.paddle.centerx - self.ball_pos[0]) < self.paddle.width:
                reward += 0.1 # Encourage being under the ball
            else:
                reward -= 0.1 # Penalize being far from a descending ball
        
        # 3. Check for termination
        terminated = False
        if self.balls_left <= 0 or self.stage_timer <= 0:
            # sfx: game_over.wav
            self.game_over = True
            terminated = True
        
        if len(self.blocks) == 0:
            self.stage += 1
            if self.stage > 3:
                # sfx: game_win.wav
                reward += 100 # Win game bonus
                self.game_over = True
                terminated = True
            else:
                # sfx: stage_clear.wav
                reward += 50 # Stage clear bonus
                self._setup_stage()
                self._setup_new_life()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball(self):
        reward = 0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_radius, self.ball_pos[1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)

        # Wall collisions
        if ball_rect.left < 0 or ball_rect.right > self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.ball_radius, min(self.WIDTH - self.ball_radius, self.ball_pos[0]))
            # sfx: wall_bounce.wav
        if ball_rect.top < 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.ball_radius, self.ball_pos[1])
            # sfx: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce.wav
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.ball_radius - 1
            
            # Influence horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] += offset * 3
            # Normalize speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.ball_speed_base
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.ball_speed_base

            # Risky bounce reward
            if abs(offset) > 0.8:
                reward += 5

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # sfx: block_hit.wav
                
                # Simple reflection logic
                # Determine if collision is more horizontal or vertical
                prev_ball_pos = (self.ball_pos[0] - self.ball_vel[0], self.ball_pos[1] - self.ball_vel[1])
                prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.ball_radius, prev_ball_pos[1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)

                if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                    self.ball_vel[1] *= -1
                if prev_ball_rect.right <= block['rect'].left or prev_ball_rect.left >= block['rect'].right:
                    self.ball_vel[0] *= -1

                block['health'] -= 1
                if block['health'] <= 0:
                    # sfx: block_break.wav
                    reward += self.stage # +1 for stage 1, +2 for stage 2, etc.
                    self._create_particles(block['rect'].center, block['color'])
                    self.blocks.remove(block)
                    self.score += 10 * self.stage
                else:
                    self.score += 1
                break # Only one block collision per frame

        # Ball lost
        if ball_rect.top > self.HEIGHT:
            # sfx: ball_lost.wav
            self.balls_left -= 1
            reward -= 10
            if self.balls_left > 0:
                self._setup_new_life()
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 25),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Blocks
        for block in self.blocks:
            r = block['rect']
            c = block['color']
            # Glow effect
            glow_rect = r.inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_alpha = 40 + 20 * (block['health'] / block['max_health'])
            # FIX: Use c.r, c.g, c.b to create a new color tuple with alpha.
            glow_color = (c.r, c.g, c.b, int(glow_alpha))
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)
            # Main block
            pygame.draw.rect(self.screen, c, r, border_radius=3)
            border_color = tuple(min(255, x + 50) for x in (c.r, c.g, c.b))
            pygame.draw.rect(self.screen, border_color, r, 1, border_radius=3)

        # Draw Paddle
        glow_paddle_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_paddle_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_paddle_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw Ball
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ball_radius + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ball_radius, self.COLOR_BALL)
        
        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25))
            # FIX: Use p['color'].r, .g, .b to create a new color tuple with alpha.
            p_color_obj = p['color']
            color = (p_color_obj.r, p_color_obj.g, p_color_obj.b, alpha)
            surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
    
    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        ball_text = self.font_medium.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - ball_text.get_width() - 10, 10))

        # Timer
        secs = self.stage_timer // self.FPS if self.stage_timer > 0 else 0
        timer_text = self.font_large.render(f"{secs}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width() / 2, 5))
        
        if self.game_over:
            if len(self.blocks) > 0:
                end_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            else:
                end_text = self.font_large.render("YOU WIN!", True, (50, 255, 50))
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "timer_seconds": self.stage_timer // self.FPS if self.stage_timer > 0 else 0,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
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
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Neon Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Human controls
        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1
        # No shift action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        if done:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000)

    env.close()