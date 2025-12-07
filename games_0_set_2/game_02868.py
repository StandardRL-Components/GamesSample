import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A procedurally generated block-breaking game where an RL agent learns to
    deflect a ball to destroy blocks using discrete directional controls.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ← and → to move the paddle."
    )

    # Short, user-facing description of the game
    game_description = (
        "A retro block-breaking game. Deflect the ball to destroy all the blocks and achieve a high score."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_PADDLE = (200, 200, 220)
    COLOR_BALL = (255, 255, 255)
    COLOR_WALL = (80, 80, 100)
    COLOR_TEXT = (240, 240, 240)
    BLOCK_COLORS = {
        5: (255, 50, 50),   # Red
        4: (255, 150, 50),  # Orange
        3: (255, 255, 50),  # Yellow
        2: (50, 255, 50),   # Green
        1: (50, 150, 255),  # Blue
    }

    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 4.0
    BALL_SPEED_INCREMENT = 0.1
    MAX_BALL_SPEED = 12.0
    
    BLOCK_COLS = 10
    BLOCK_ROWS = 5
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 6
    BLOCK_AREA_TOP = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES

        # Paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        self.ball_speed = self.INITIAL_BALL_SPEED
        # FIX: low must be less than high. Swapped arguments.
        angle = self.np_random.uniform(-math.pi * 0.6, -math.pi * 0.4)
        self.ball_vel = [self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)]
        
        # Blocks
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                hp = self.BLOCK_ROWS - r
                color = self.BLOCK_COLORS[hp]
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "hp": hp, "color": color})

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            # Unpack action
            movement = action[0]

            # 1. Update paddle position based on action
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            self.paddle.clamp_ip(self.screen.get_rect())

            # 2. Update ball position
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # 3. Handle collisions
            # Walls
            if ball_rect.left <= 0:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = self.BALL_RADIUS
            if ball_rect.right >= self.SCREEN_WIDTH:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            if ball_rect.top <= 0:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = self.BALL_RADIUS

            # Paddle
            if self.ball_vel[1] > 0 and ball_rect.colliderect(self.paddle):
                # Reverse Y velocity
                self.ball_vel[1] *= -1
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
                
                # Change X velocity based on hit location
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += offset * 2.5
                
                # Normalize to maintain speed
                self._normalize_ball_velocity()
                # sfx: paddle_hit

            # Blocks
            hit_block_this_frame = False
            for block in reversed(self.blocks):
                if not hit_block_this_frame and ball_rect.colliderect(block["rect"]):
                    hit_block_this_frame = True
                    reward += 0.1
                    
                    # Determine bounce direction
                    self._handle_ball_block_collision(ball_rect, block["rect"])
                    
                    block["hp"] -= 1
                    if block["hp"] <= 0:
                        reward += 1.0
                        self.score += 10
                        self._create_particles(block["rect"].center, block["color"])
                        self.blocks.remove(block)
                        self.ball_speed = min(self.MAX_BALL_SPEED, self.ball_speed + self.BALL_SPEED_INCREMENT)
                        self._normalize_ball_velocity()
                        # sfx: block_destroy
                    else:
                        block["color"] = self.BLOCK_COLORS[block["hp"]]
                        # sfx: block_hit
                    break

            # Bottom wall (lose life)
            if ball_rect.top >= self.SCREEN_HEIGHT:
                self.lives -= 1
                reward -= 0.01
                # sfx: lose_life
                if self.lives <= 0:
                    self.game_over = True
                    reward -= 100.0
                else:
                    # Reset ball
                    self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                    self.ball_speed = self.INITIAL_BALL_SPEED
                    # FIX: low must be less than high. Swapped arguments.
                    angle = self.np_random.uniform(-math.pi * 0.6, -math.pi * 0.4)
                    self.ball_vel = [self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)]

        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination conditions
        terminated = self.game_over
        if not self.blocks and not terminated:
            self.game_over = True
            terminated = True
            reward += 100.0
            self.score += 1000 # Win bonus
        
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True # Environment is episodic, so this is termination
            truncated = True # But it's also due to a time limit

        # Gymnasium API expects terminated OR truncated, not both.
        # Let's prioritize termination due to game over conditions.
        if self.game_over:
            truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            pos, radius, color = p[0], p[2], p[4]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)
        
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball (with glow)
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 100 + i * 30, 12, 25, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)
            
        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (100, 255, 100) if not self.blocks else (255, 100, 100)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def _normalize_ball_velocity(self):
        current_magnitude = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if current_magnitude > 0:
            scale = self.ball_speed / current_magnitude
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

    def _handle_ball_block_collision(self, ball_rect, block_rect):
        # A simple but effective collision response
        overlap = ball_rect.clip(block_rect)
        
        if overlap.width < overlap.height:
            # Horizontal collision
            self.ball_vel[0] *= -1
            self.ball_pos[0] += self.ball_vel[0] # Push out
        else:
            # Vertical collision
            self.ball_vel[1] *= -1
            self.ball_pos[1] += self.ball_vel[1] # Push out

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(15, 30)
            # pos, vel, radius, life, color
            self.particles.append([[pos[0], pos[1]], vel, radius, life, color])
            
    def _update_particles(self):
        for p in reversed(self.particles):
            p[0][0] += p[1][0] # Update pos x
            p[0][1] += p[1][1] # Update pos y
            p[1][1] += 0.1     # Gravity
            p[3] -= 1          # Decrement life
            p[2] *= 0.97       # Shrink
            if p[3] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Note: This is for human play and debugging, not for RL training.
    
    # Override some settings for human play
    GameEnv.auto_advance = False 
    
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Setup Pygame display window
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    # Convert observation for display
    def display_obs(observation):
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose and then create a surface
        disp_surf = pygame.surfarray.make_surface(np.transpose(observation, (1, 0, 2)))
        screen.blit(disp_surf, (0, 0))
        pygame.display.flip()

    while not (terminated or truncated):
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        display_obs(obs)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if info.get('game_over'):
            # Pause on game over to show message, wait for 'r' to restart
            pass
        
        clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()