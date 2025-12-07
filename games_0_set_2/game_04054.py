import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Break all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Control a paddle to bounce a ball and break all the blocks. "
        "Earn bonus points for consecutive breaks, but lose a life if you miss the ball."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BOUNDARY = (50, 50, 100)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 50, 50), (255, 150, 50), (50, 255, 50),
        (50, 150, 255), (200, 50, 200)
    ]
    
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    MAX_BALL_SPEED = 10.0
    
    MAX_STEPS = 10000
    INITIAL_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.consecutive_breaks = 0
        self.total_blocks_destroyed = 0
        self.initial_block_count = 0
        self.last_paddle_x = 0

        # Note: self.reset() is called by the Gym wrapper, no need to call it here.
        # self.reset()

        # self.validate_implementation() # This can be useful for debugging
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.last_paddle_x = self.paddle.x
        
        self._reset_ball()

        self.blocks = []
        block_rows = 5
        block_cols = 10
        block_width = 58
        block_height = 20
        start_x = (self.WIDTH - (block_cols * (block_width + 5)) + 5) / 2
        start_y = 50
        for i in range(block_rows):
            for j in range(block_cols):
                block_rect = pygame.Rect(
                    start_x + j * (block_width + 5),
                    start_y + i * (block_height + 5),
                    block_width,
                    block_height
                )
                block_info = {
                    'rect': block_rect,
                    'color': self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                }
                self.blocks.append(block_info)
        self.initial_block_count = len(self.blocks)
        
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.consecutive_breaks = 0
        self.total_blocks_destroyed = 0
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        speed = self._get_current_ball_speed()
        self.ball_vel = [speed * math.cos(angle), speed * math.sin(angle)]

    def _get_current_ball_speed(self):
        speed_increase_factor = self.total_blocks_destroyed // 5
        speed = self.INITIAL_BALL_SPEED + speed_increase_factor * 0.5
        return min(speed, self.MAX_BALL_SPEED)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement = action[0]

        # 1. Handle Input
        self._handle_input(movement)
        
        # 2. Update Physics
        reward += self._update_physics()

        # 3. Update Particles
        self._update_particles()
        
        # 4. Update Step Counter
        self.steps += 1
        
        # 5. Check Termination Conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.win:
                reward += 100  # Win bonus
            elif self.lives <= 0:
                reward -= 100  # Loss penalty
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        self.last_paddle_x = self.paddle.x
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

    def _update_physics(self):
        step_reward = 0
        
        # Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = np.clip(ball_rect.left, 0, self.WIDTH - ball_rect.width)
            self.ball_pos[0] = ball_rect.centerx
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            step_reward += 0.1  # Reward for hitting the ball
            self.consecutive_breaks = 0 # Reset combo
            
            offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
            angle = math.pi/2 + offset * (math.pi / 3) 
            
            speed = self._get_current_ball_speed()
            self.ball_vel[0] = speed * -math.cos(angle)
            self.ball_vel[1] = speed * -math.sin(angle)

            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS 
            self._create_particles(ball_rect.midbottom, self.COLOR_PADDLE, 15)

        # Block collisions
        block_rects = [b['rect'] for b in self.blocks]
        hit_block_index = ball_rect.collidelist(block_rects)
        if hit_block_index != -1:
            hit_block_info = self.blocks[hit_block_index]
            block_to_remove_rect = hit_block_info['rect']
            block_color = hit_block_info['color']
            
            step_reward += 1.0 # Reward for breaking a block
            self.consecutive_breaks += 1
            if self.consecutive_breaks > 1:
                step_reward += 0.5 # Combo bonus

            self.score += 10 + (self.consecutive_breaks - 1) * 5
            self.total_blocks_destroyed += 1

            self._create_particles(block_to_remove_rect.center, block_color, 30)
            
            self.ball_vel[1] *= -1

            self.blocks.pop(hit_block_index)

        # Miss (ball below paddle)
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            step_reward -= 5 # Penalty for losing a life
            self.consecutive_breaks = 0
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
                self.win = False

        # Penalty for not moving when ball is approaching
        ball_approaching = self.ball_vel[1] > 0 and self.ball_pos[1] > self.HEIGHT / 2
        paddle_moved = self.paddle.x != self.last_paddle_x
        if ball_approaching and not paddle_moved:
            step_reward -= 0.02

        return step_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0] # pos x
            p[0][1] += p[1][1] # pos y
            p[2] -= 1 # timer

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            timer = self.np_random.integers(15, 30)
            self.particles.append([list(pos), vel, timer, color])

    def _check_termination(self):
        if not self.blocks:
            self.win = True
            return True
        if self.lives <= 0:
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw blocks
        for block_info in self.blocks:
            pygame.draw.rect(self.screen, block_info['color'], block_info['rect'])

        # Draw particles
        for pos, vel, timer, color in self.particles:
            alpha = max(0, min(255, int(255 * (timer / 30.0))))
            size = max(1, int(3 * (timer / 30.0)))
            # Pygame doesn't handle alpha well on surfaces without SRCALPHA flag, so we simulate it
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(pos[0]) - size, int(pos[1]) - size))


        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_radius = int(self.BALL_RADIUS * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, glow_radius, (*self.COLOR_BALL, 40))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, glow_radius, (*self.COLOR_BALL, 60))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_width = self.PADDLE_WIDTH / 3
        life_icon_height = self.PADDLE_HEIGHT / 2
        for i in range(self.lives):
            x = self.WIDTH - (i + 1) * (life_icon_width + 5) - 5
            y = 10
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, (x, y, life_icon_width, life_icon_height), border_radius=2)
            
        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surface = self.font_game_over.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = text_rect.inflate(40, 40)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((20, 20, 30, 180))
            self.screen.blit(bg_surface, bg_rect.topleft)
            self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
            "win": self.win
        }

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

# Example of how to run the environment
if __name__ == '__main__':
    # --- For verification (headless) ---
    print("Running headless verification...")
    env = GameEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    print("Headless verification passed.")
    env.close()

    # --- To play with keyboard (requires a display) ---
    try:
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        clock = pygame.time.Clock()

        done = False
        print("\nStarting interactive game. Use Left/Right arrow keys. Press R to reset, Q to quit.")
        while not done:
            movement = 0 # No-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            action = [movement, 0, 0] # Movement, space, shift
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

            clock.tick(60)
        
        print(f"Game Over! Final Score: {info['score']}")
        env.close()

    except (pygame.error, ImportError) as e:
        print("\nPygame display not available or error occurred. Skipping interactive test.")
        print(f"Error: {e}")