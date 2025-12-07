
# Generated: 2025-08-28T02:54:10.113170
# Source Brief: brief_04602.md
# Brief Index: 4602

        
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
        "Minimalist block breaker. Clear all blocks on each level before the timer runs out or you lose all your lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 5400 # 60 seconds/level * 3 levels * 30 FPS

    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (76, 175, 80), 
        (33, 150, 243), (156, 39, 176), (233, 30, 99)
    ]

    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    PADDLE_SPIN_FACTOR = 4

    BALL_RADIUS = 7
    
    MAX_LEVELS = 3
    LEVEL_TIME = 1800 # 60 seconds * 30 FPS
    SOFTLOCK_THRESHOLD = 300 # 10 seconds * 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.block_colors = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.level = 0
        self.level_timer = 0
        self.steps_since_last_hit = 0
        self.win = False
        self.game_over = False
        
        # Initialize state
        self.reset()

        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.level = 1
        self.win = False
        self.game_over = False
        self.particles = []

        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._setup_level(self.level)
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self, level_num):
        self.blocks = []
        self.block_colors = []
        self.level_timer = self.LEVEL_TIME
        self.steps_since_last_hit = 0

        # Reset ball
        self.ball_attached = True
        ball_speed_base = 5.0 + (level_num - 1) * 0.5
        self.ball_vel = [0, -ball_speed_base]
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Generate blocks
        block_counts = [50, 60, 70]
        num_blocks = block_counts[level_num - 1]
        
        cols = 10
        rows = num_blocks // cols
        block_width = (self.SCREEN_WIDTH - 20) // cols
        block_height = 20
        
        for i in range(num_blocks):
            row = i // cols
            col = i % cols
            x = 10 + col * block_width
            y = 50 + row * (block_height + 5)
            block = pygame.Rect(x, y, block_width - 5, block_height)
            self.blocks.append(block)
            self.block_colors.append(random.choice(self.BLOCK_COLORS))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if space_held and self.ball_attached:
            self.ball_attached = False
            # Sound effect placeholder: # Play launch sound
            initial_angle = self.np_random.uniform(-0.5, 0.5)
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel = [speed * math.sin(initial_angle), -speed * math.cos(initial_angle)]

        # --- Update Game State ---
        self.steps += 1
        self.level_timer -= 1
        self._update_particles()
        
        if self.ball_attached:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            self.steps_since_last_hit += 1
            reward += self._update_ball()

        # --- Check Game State Transitions ---
        if not self.blocks:
            reward += 10 # Level clear bonus
            # Sound effect placeholder: # Play level up sound
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.win = True
                self.game_over = True
                reward += 100 # Win game bonus
            else:
                self._setup_level(self.level)
        
        if self.level_timer <= 0 or self.steps_since_last_hit > self.SOFTLOCK_THRESHOLD:
            reward += self._lose_life()
            if self.level_timer <= 0:
                self.level_timer = self.LEVEL_TIME

        terminated = self.lives <= 0 or self.win or self.steps >= self.MAX_EPISODE_STEPS
        if terminated and not self.game_over:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0.0
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            self.ball_vel[0] *= -1
            # Sound effect placeholder: # Play wall hit sound
        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_vel[1] *= -1
            # Sound effect placeholder: # Play wall hit sound

        # Bottom wall (lose life)
        if self.ball.top >= self.SCREEN_HEIGHT:
            reward += self._lose_life()
            return reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * self.PADDLE_SPIN_FACTOR
            
            # Normalize speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            base_speed = 5.0 + (self.level - 1) * 0.5
            self.ball_vel[0] = (self.ball_vel[0] / speed) * base_speed
            self.ball_vel[1] = (self.ball_vel[1] / speed) * base_speed

            reward += 0.1 if abs(offset) > 0.5 else -0.01
            self.steps_since_last_hit = 0
            # Sound effect placeholder: # Play paddle hit sound

        # Block collisions
        collided_idx = self.ball.collidelist(self.blocks)
        if collided_idx != -1:
            block = self.blocks.pop(collided_idx)
            color = self.block_colors.pop(collided_idx)
            
            self._create_particles(block.center, color)
            
            # Determine collision side
            # A simple but effective model: reverse vertical velocity
            self.ball_vel[1] *= -1

            self.score += 10
            reward += 1
            self.steps_since_last_hit = 0
            # Sound effect placeholder: # Play block break sound

        return reward

    def _lose_life(self):
        self.lives -= 1
        self.ball_attached = True
        self.steps_since_last_hit = 0
        # Sound effect placeholder: # Play lose life sound
        if self.lives <= 0:
            self.game_over = True
        return -10

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with anti-aliasing
        x, y, r = int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS
        pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font.render(f"Lives: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Timer
        time_left = max(0, self.level_timer // self.FPS)
        timer_text = self.font.render(f"Time: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if not self.win else "YOU WIN!"
            msg_color = (255, 50, 50) if not self.win else (50, 255, 50)
            msg_surf = self.font.render(msg, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
            "win": self.win,
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

# Example usage to test the environment
if __name__ == '__main__':
    import time
    
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    # For human mode, we need a screen to display to
    if render_mode == "human":
        pygame.display.set_caption("Block Breaker")
        real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()

    terminated = False
    total_reward = 0
    
    # --- Human Controls ---
    # This block allows a human to play the game
    # The agent will just take random actions if this is disabled
    use_human_player = True
    if use_human_player and render_mode != "human":
        print("Warning: Human player requires render_mode='human'")
        use_human_player = False
    
    keys_pressed = {
        "left": False,
        "right": False,
        "space": False,
    }

    while not terminated:
        action = env.action_space.sample() # Default to random action

        if use_human_player:
            # Poll for Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: keys_pressed["left"] = True
                    if event.key == pygame.K_RIGHT: keys_pressed["right"] = True
                    if event.key == pygame.K_SPACE: keys_pressed["space"] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT: keys_pressed["left"] = False
                    if event.key == pygame.K_RIGHT: keys_pressed["right"] = False
                    if event.key == pygame.K_SPACE: keys_pressed["space"] = False
            
            # Map keys to action space
            movement = 0
            if keys_pressed["left"]: movement = 3
            elif keys_pressed["right"]: movement = 4
            
            space = 1 if keys_pressed["space"] else 0
            shift = 0 # Not used
            
            action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # The environment's observation is already rendered, we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)
            
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(2)
            obs, info = env.reset()
            total_reward = 0
            terminated = False # Remove this line to play only one episode

    env.close()