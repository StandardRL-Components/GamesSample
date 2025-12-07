
# Generated: 2025-08-27T22:29:09.901646
# Source Brief: brief_03142.md
# Brief Index: 3142

        
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
        "Bounce a ball to destroy all blocks in this fast-paced arcade game. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 6.0
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 20
        self.WALL_THICKNESS = 10
        self.MAX_STEPS = 1000
        self.INITIAL_HITS = 3
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PADDLE = (200, 200, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (150, 150, 180)
        self.BLOCK_COLORS = {
            3: (255, 50, 50),   # Red
            2: (50, 150, 255),  # Blue
            1: (50, 255, 50),   # Green
        }
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.hits_left = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.hits_left = self.INITIAL_HITS
        self.game_over = False
        
        # Paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball
        self.ball_launched = False
        self._reset_ball()
        
        # Blocks
        self._generate_blocks()
        
        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            space_pressed = action[1] == 1
            
            # Handle input and calculate related rewards
            reward += self._handle_input(movement, space_pressed)
            
            # Update game state and calculate event rewards
            reward += self._update_game_state()

        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _generate_blocks(self):
        self.blocks = []
        block_rows = 5
        block_cols = 10
        gap = 2
        total_block_width = block_cols * self.BLOCK_WIDTH + (block_cols - 1) * gap
        x_offset = (self.WIDTH - total_block_width) / 2
        y_offset = 50
        
        for i in range(block_rows):
            for j in range(block_cols):
                points = 1
                if i < 2: points = 3  # Top two rows are red
                elif i < 4: points = 2 # Middle two rows are blue
                
                block_x = x_offset + j * (self.BLOCK_WIDTH + gap)
                block_y = y_offset + i * (self.BLOCK_HEIGHT + gap)
                rect = pygame.Rect(block_x, block_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": rect, "points": points, "color": self.BLOCK_COLORS[points]})

    def _handle_input(self, movement, space_pressed):
        reward = 0
        
        # Predictive reward for paddle movement
        if self.ball_launched and self.ball_vel.y > 0 and self.ball_pos.y > self.HEIGHT / 2:
            try:
                time_to_impact = (self.paddle_rect.top - self.ball_pos.y) / self.ball_vel.y
                if time_to_impact > 0:
                    projected_x = self.ball_pos.x + self.ball_vel.x * time_to_impact
                    # Simple wall bounce prediction
                    while projected_x < self.WALL_THICKNESS or projected_x > self.WIDTH - self.WALL_THICKNESS:
                        if projected_x < self.WALL_THICKNESS: projected_x = 2 * self.WALL_THICKNESS - projected_x
                        if projected_x > self.WIDTH - self.WALL_THICKNESS: projected_x = 2 * (self.WIDTH - self.WALL_THICKNESS) - projected_x
                    
                    current_dist = abs(self.paddle_rect.centerx - projected_x)
                    next_paddle_x = self.paddle_rect.x
                    if movement == 3: next_paddle_x -= self.PADDLE_SPEED
                    elif movement == 4: next_paddle_x += self.PADDLE_SPEED
                    
                    next_dist = abs((next_paddle_x + self.PADDLE_WIDTH / 2) - projected_x)
                    if next_dist > current_dist and movement in [3, 4]:
                        reward -= 0.2
            except ZeroDivisionError:
                pass

        # Move paddle
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(self.WALL_THICKNESS, min(self.paddle_rect.x, self.WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))
        
        # Launch ball
        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            launch_angle = (self.np_random.random() * 0.8 - 0.4) * math.pi # -72 to 72 degrees
            self.ball_vel = pygame.Vector2(math.sin(launch_angle), -math.cos(launch_angle)) * self.BALL_SPEED_INITIAL
            # sfx: launch_sound
            
        # If ball not launched, it follows the paddle
        if not self.ball_launched:
            self.ball_pos.x = self.paddle_rect.centerx
            
        return reward
        
    def _update_game_state(self):
        reward = 0
        if not self.ball_launched:
            return reward

        # Move ball
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= self.WALL_THICKNESS or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.ball_pos.x, self.WALL_THICKNESS + self.BALL_RADIUS)
            self.ball_pos.x = min(self.ball_pos.x, self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS)
            # sfx: wall_bounce_sound
        if self.ball_pos.y - self.BALL_RADIUS <= self.WALL_THICKNESS:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.ball_pos.y, self.WALL_THICKNESS + self.BALL_RADIUS)
            # sfx: wall_bounce_sound

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            reward += 0.1
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
            self.ball_vel.y *= -1
            
            # Change angle based on hit location
            offset = (self.ball_pos.x - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 4.0
            
            # Maintain constant speed
            if self.ball_vel.length() > 0:
                self.ball_vel = self.ball_vel.normalize() * self.BALL_SPEED_INITIAL
            # sfx: paddle_hit_sound

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                reward += block["points"]
                self.score += block["points"]
                self._create_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                self.ball_vel.y *= -1
                # sfx: block_break_sound
                break
        
        # Ball miss
        if self.ball_pos.y - self.BALL_RADIUS > self.HEIGHT:
            self.hits_left -= 1
            if self.hits_left <= 0:
                self.game_over = True
                reward -= 100
                # sfx: lose_sound
            else:
                self._reset_ball()
                # sfx: miss_sound

        # Win condition
        if not self.blocks:
            self.game_over = True
            reward += 100
            # sfx: win_sound
            
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
                
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Draw ball trail
        if self.ball_launched:
            p1 = self.ball_pos - self.ball_vel * 0.5
            p2 = self.ball_pos - self.ball_vel
            pygame.gfxdraw.filled_circle(self.screen, int(p1.x), int(p1.y), self.BALL_RADIUS-1, (180, 180, 220, 100))
            pygame.gfxdraw.filled_circle(self.screen, int(p2.x), int(p2.y), self.BALL_RADIUS-2, (150, 150, 200, 50))
        
        # Draw ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30.0))))
            color = p["color"] + (alpha,)
            size = max(1, int(p["lifespan"] / 10))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.HEIGHT - 30))
        
        # Hits
        hits_text = self.font_main.render(f"HITS: {self.hits_left}", True, (255, 255, 255))
        self.screen.blit(hits_text, (self.WIDTH - hits_text.get_width() - self.WALL_THICKNESS - 10, self.HEIGHT - 30))
        
        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (50, 255, 50) if not self.blocks else (255, 50, 50)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits_left": self.hits_left,
            "blocks_left": len(self.blocks),
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
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        # Control frame rate
        clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    env.close()