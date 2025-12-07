
# Generated: 2025-08-27T19:13:55.143904
# Source Brief: brief_02088.md
# Brief Index: 2088

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball to destroy all blocks in a fast-paced, top-down arcade block breaker. Clear 3 stages to win."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Colors ---
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PADDLE = (0, 220, 220)
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_BALL_GLOW = (255, 100, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            1: (0, 255, 100),
            2: (255, 255, 0),
            3: (255, 50, 50),
        }
        
        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 28)
            self.font_large = pygame.font.Font(None, 36)

        # --- Game Constants ---
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.MAX_LIVES = 3
        self.MAX_STAGES = 3
        self.STAGE_TIME_SECONDS = 60
        self.FPS = 30 # Physics is tuned to this framerate
        self.MAX_STEPS = self.STAGE_TIME_SECONDS * self.FPS * self.MAX_STAGES
        
        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed_magnitude = None
        self.ball_launched = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.stage_timer = 0
        self.game_over = False
        self.steps = 0
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.stage = 1
        self.game_over = False
        self.particles = []
        
        self._setup_stage(self.stage)
        self._reset_ball_and_paddle()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement in [3, 4]: # Left or Right
            reward -= 0.02 # Small penalty for moving
            if movement == 3: # Left
                self.paddle_rect.x -= self.PADDLE_SPEED
            elif movement == 4: # Right
                self.paddle_rect.x += self.PADDLE_SPEED
            self.paddle_rect.clamp_ip(self.screen.get_rect())
        
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # Launch with a slight random angle
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = [
                self.ball_speed_magnitude * math.sin(angle),
                -self.ball_speed_magnitude * math.cos(angle)
            ]
            # Sound effect placeholder: # LAUNCH_SOUND

        # --- Update Game Logic ---
        self.steps += 1
        self.stage_timer -= 1
        
        if not self.ball_launched:
            # Ball follows paddle
            self.ball_pos[0] = self.paddle_rect.centerx
        else:
            # Move ball
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            
            # --- Handle Collisions & Rewards ---
            reward += self._handle_collisions()

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Stage Clear ---
        if not self.blocks and not self.game_over:
            reward += 100.0 # Stage clear bonus
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                reward += 300.0 # Game win bonus
            else:
                self._setup_stage(self.stage)
                self._reset_ball_and_paddle()
                # Sound effect placeholder: # STAGE_CLEAR_SOUND

        # --- Check Termination Conditions ---
        if self.lives <= 0 or self.stage_timer <= 0:
            self.game_over = True
            reward -= 100.0 # Game over penalty
        
        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _setup_stage(self, stage_num):
        self.blocks = []
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        
        # Increase ball speed per stage
        self.ball_speed_magnitude = 7 + (stage_num - 1) * 0.5
        
        num_rows = 4 + stage_num
        num_cols = 10
        block_width = self.width // num_cols
        block_height = 20
        y_offset = 60

        for r in range(num_rows):
            for c in range(num_cols):
                # Simple checkerboard pattern for variety
                if (r + c) % 2 == 0:
                    continue
                
                hp = min(self.np_random.integers(1, stage_num + 2), 3)
                
                block_rect = pygame.Rect(
                    c * block_width + 1,
                    r * block_height + y_offset,
                    block_width - 2,
                    block_height - 2
                )
                self.blocks.append({"rect": block_rect, "hp": hp, "initial_hp": hp})

    def _reset_ball_and_paddle(self):
        self.paddle_rect = pygame.Rect(
            (self.width - self.PADDLE_WIDTH) / 2,
            self.height - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def _handle_collisions(self):
        reward = 0.0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        # Walls
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.width - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.width - self.BALL_RADIUS)
            # Sound effect placeholder: # WALL_BOUNCE_SOUND
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.height - self.BALL_RADIUS)
            # Sound effect placeholder: # WALL_BOUNCE_SOUND

        # Bottom (lose life)
        if self.ball_pos[1] >= self.height + self.BALL_RADIUS:
            self.lives -= 1
            reward -= 1.0
            if self.lives > 0:
                self._reset_ball_and_paddle()
                # Sound effect placeholder: # LOSE_LIFE_SOUND
            return reward # Stop processing collisions for this frame

        # Paddle
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # Sound effect placeholder: # PADDLE_HIT_SOUND
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
            
            # Calculate influence on horizontal velocity
            offset = (self.ball_pos[0] - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.ball_speed_magnitude * offset
            self.ball_vel[1] *= -1
            
            # Normalize to maintain speed
            current_speed = math.hypot(*self.ball_vel)
            if current_speed > 0:
                scale = self.ball_speed_magnitude / current_speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale

        # Blocks
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # Sound effect placeholder: # BLOCK_HIT_SOUND
                reward += 0.1 # Hit bonus

                # Simple collision response
                # Find overlap
                dx = (ball_rect.centerx - block["rect"].centerx) / block["rect"].width
                dy = (ball_rect.centery - block["rect"].centery) / block["rect"].height
                
                if abs(dx) > abs(dy):
                    self.ball_vel[0] *= -1
                    # Push out
                    self.ball_pos[0] += np.sign(self.ball_vel[0]) * 2
                else:
                    self.ball_vel[1] *= -1
                    # Push out
                    self.ball_pos[1] += np.sign(self.ball_vel[1]) * 2

                block["hp"] -= 1
                if block["hp"] <= 0:
                    reward += block["initial_hp"] # Destruction bonus
                    self.score += block["initial_hp"] * 10
                    self._create_particles(block["rect"].center, self.BLOCK_COLORS[block["initial_hp"]])
                    self.blocks.remove(block)
                    # Sound effect placeholder: # BLOCK_DESTROY_SOUND
                
                # Only handle one block collision per frame for simplicity
                break 
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # --- Draw Background Gradient ---
        for y in range(self.height):
            interp = y / self.height
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw Blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[block["hp"]]
            pygame.draw.rect(self.screen, color, block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block["rect"], width=2, border_radius=3)

        # Draw Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)
        
        # Draw Ball with glow
        if self.ball_pos:
            x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
            # Glow effect
            for i in range(self.BALL_RADIUS, 0, -2):
                alpha = 40 * (1 - i / self.BALL_RADIUS)
                pygame.gfxdraw.filled_circle(self.screen, x, y, i + 5, (*self.COLOR_BALL_GLOW, alpha))
            # Ball itself
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 20))
            color = (*p["color"], alpha)
            size = max(1, int(p["lifespan"] / 5))
            pygame.draw.circle(self.screen, p["color"], p["pos"], size)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Stage
        stage_surf = self.font_large.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.width // 2 - stage_surf.get_width() // 2, 10))

        # Timer
        time_left = max(0, self.stage_timer // self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else (255, 100, 100)
        time_surf = self.font_main.render(f"TIME: {time_left}", True, time_color)
        self.screen.blit(time_surf, (self.width // 2 - time_surf.get_width() // 2, 40))

        # Lives
        heart_char = "♥"
        lives_str = " ".join([heart_char] * self.lives)
        lives_surf = self.font_large.render(lives_str, True, (255, 80, 80))
        self.screen.blit(lives_surf, (self.width - lives_surf.get_width() - 15, 8))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Control Setup ---
    # Action map: [movement, space, shift]
    # movement: 0=none, 3=left, 4=right
    # space: 0=released, 1=held
    # shift: 0=released, 1=held
    
    # Pygame window for human play
    render_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Block Breaker")
    
    total_reward = 0
    
    while not terminated:
        # --- Get Human Input ---
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        action = [0, 0, 0] # [movement, space, shift]
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Check for Quit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Frame Rate ---
        env.clock.tick(env.FPS)

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Survived {info['steps']} steps.")
    
    env.close()