
# Generated: 2025-08-28T00:10:11.444442
# Source Brief: brief_03708.md
# Brief Index: 3708

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic block breaker. Clear the screen of all blocks by bouncing the ball off your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    BALL_SPEED = 7
    MAX_STEPS = 2500

    # Colors (Retro Neon Theme)
    COLOR_BG_START = (10, 0, 30)
    COLOR_BG_END = (30, 0, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (0, 255, 200)
    COLOR_BALL_GLOW = (0, 255, 200, 50)
    BLOCK_COLORS = [
        (255, 0, 128), (200, 0, 255), (0, 128, 255), 
        (0, 255, 128), (255, 255, 0)
    ]
    COLOR_TEXT = (220, 220, 220)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.total_blocks = 0
        
        # Create a persistent gradient background surface
        self.bg_surface = self._create_gradient_background()

        self.reset()
        
        # Self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        self.paddle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2 - self.PADDLE_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.ball_attached = True
        self._reset_ball()
        
        self._generate_blocks()
        self.total_blocks = len(self.blocks)
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        gap = 5
        rows = 5
        cols = self.SCREEN_WIDTH // (block_width + gap)
        
        for r in range(rows):
            for c in range(cols):
                # Use RNG to decide if a block should be placed
                if self.np_random.random() > 0.2:
                    x = c * (block_width + gap) + gap * 2
                    y = r * (block_height + gap) + 50
                    rect = pygame.Rect(x, y, block_width, block_height)
                    color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                    self.blocks.append({'rect': rect, 'color': color, 'points': (rows - r) * 1})

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(
            self.paddle_pos.x + self.PADDLE_WIDTH / 2,
            self.paddle_pos.y - self.BALL_RADIUS
        )
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_attached = True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, we care about the press event
        
        reward = -0.001  # Small penalty for time to encourage efficiency
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos.x += self.PADDLE_SPEED
        
        self.paddle_pos.x = np.clip(self.paddle_pos.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if self.ball_attached:
            self.ball_pos.x = self.paddle_pos.x + self.PADDLE_WIDTH / 2
            if space_pressed:
                self.ball_attached = False
                # Launch with a slight random angle
                angle = self.np_random.uniform(-math.pi * 0.1, math.pi * 0.1)
                self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED
                # Sound: Ball launch
        
        # --- Update Game Logic ---
        if not self.ball_attached:
            self.ball_pos += self.ball_vel

            # Wall collisions
            if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                self.ball_vel.x *= -1
                self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
                # Sound: Wall bounce
            if self.ball_pos.y <= self.BALL_RADIUS:
                self.ball_vel.y *= -1
                self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.SCREEN_HEIGHT)
                # Sound: Wall bounce

            # Bottom wall (lose ball)
            if self.ball_pos.y >= self.SCREEN_HEIGHT + self.BALL_RADIUS:
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
                    reward -= 10 # Large penalty for losing

            # Paddle collision
            paddle_rect = pygame.Rect(self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
            if paddle_rect.colliderect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2) and self.ball_vel.y > 0:
                reward += 0.1 # Reward for hitting the ball
                
                # Calculate hit position and adjust ball velocity
                hit_offset = (self.ball_pos.x - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel.x = self.BALL_SPEED * hit_offset
                self.ball_vel.y *= -1
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.BALL_SPEED

                # Risky/Safe shot rewards
                if abs(hit_offset) > 0.7:
                    reward += 0.5 # Bonus for risky edge shot
                elif abs(hit_offset) < 0.2:
                    reward -= 0.2 # Penalty for safe center shot
                
                self.ball_pos.y = self.paddle_pos.y - self.BALL_RADIUS
                # Sound: Paddle bounce
            
            # Block collisions
            hit_block = None
            for i, block_data in enumerate(self.blocks):
                if block_data['rect'].colliderect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2):
                    hit_block = i
                    break
            
            if hit_block is not None:
                block_data = self.blocks.pop(hit_block)
                reward += block_data['points']
                self.score += block_data['points']
                
                # Create particle explosion
                self._create_particles(block_data['rect'].center, block_data['color'])

                # Bounce logic
                self.ball_vel.y *= -1 # Simple bounce is fine for this game
                # Sound: Block break
        
        # Update particles
        self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self.game_over or (len(self.blocks) == 0) or (self.steps >= self.MAX_STEPS)
        if terminated and not self.game_over:
            if len(self.blocks) == 0:
                reward += 20 # Large bonus for clearing the level
                self.score += 50 # Score bonus
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data['color'], block_data['rect'])
            pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in block_data['color']), block_data['rect'], 2)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, (self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT), border_radius=3)
        
        # Render ball with glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                color = (
                    int(p['color'][0] * life_ratio),
                    int(p['color'][1] * life_ratio),
                    int(p['color'][2] * life_ratio)
                )
                pygame.draw.circle(self.screen, color, p['pos'], radius)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            x = self.SCREEN_WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.aacircle(self.screen, x, 22, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, 22, self.BALL_RADIUS, self.COLOR_BALL)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "LEVEL CLEARED!" if len(self.blocks) == 0 else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11" or "windows" or "mac" etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    terminated = False
    
    # --- Human Controls ---
    # Convert keyboard presses to the MultiDiscrete action space
    def get_action_from_keys(keys):
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        return [movement, space, shift]

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        action = get_action_from_keys(keys)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Limit to 60 FPS for human play

    env.close()
    print(f"Game Over. Final Info: {info}")