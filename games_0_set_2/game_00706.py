
# Generated: 2025-08-27T14:30:40.813029
# Source Brief: brief_00706.md
# Brief Index: 706

        
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
        "A retro block-breaking game. Use the paddle to bounce the ball and destroy all the colored blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font = pygame.font.Font(None, 36)
        self.game_over_font = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (0, 200, 100),   # Green
            20: (0, 150, 255),   # Blue
            30: (255, 80, 80)    # Red
        }

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6.0
        self.MAX_BALL_SPEED = 12.0
        self.WALL_THICKNESS = 10
        self.MAX_STEPS = 10000
        self.INITIAL_BALLS = 5

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.ball_held = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.balls_left = None
        self.steps = None
        self.blocks_destroyed_count = None
        self.game_over = None
        self.win_state = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.balls_left = self.INITIAL_BALLS
        self.game_over = False
        self.win_state = False
        self.blocks_destroyed_count = 0

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - self.WALL_THICKNESS,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()

        self.blocks = self._generate_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_held = True
        self.ball_speed = self.INITIAL_BALL_SPEED + 0.1 * (self.blocks_destroyed_count // 20)
        self.ball_speed = min(self.ball_speed, self.MAX_BALL_SPEED)
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _generate_blocks(self):
        blocks = []
        block_width = 58
        block_height = 20
        rows = 5
        cols = 10
        top_offset = 50
        
        for i in range(rows):
            for j in range(cols):
                points = [10, 10, 20, 20, 30][i]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    self.WALL_THICKNESS + 2 + j * (block_width + 2),
                    top_offset + i * (block_height + 2),
                    block_width,
                    block_height
                )
                blocks.append({"rect": rect, "color": color, "points": points})
        return blocks

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        self.steps += 1

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_pressed = action[1] == 1

            # 1. Update paddle position
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(self.WALL_THICKNESS, self.paddle.x)
            self.paddle.x = min(self.WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH, self.paddle.x)

            # 2. Handle ball launching
            if self.ball_held:
                self.ball_pos.x = self.paddle.centerx
                if space_pressed:
                    # Sound: Ball Launch
                    self.ball_held = False
                    self.ball_vel = pygame.Vector2(random.uniform(-0.5, 0.5), -1)
                    self.ball_vel = self.ball_vel.normalize() * self.ball_speed
            else:
                # 3. Update ball position
                self.ball_pos += self.ball_vel

            # 4. Handle collisions
            reward += self._handle_collisions()

        # 5. Update particles
        self._update_particles()

        # 6. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win_state:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= self.WALL_THICKNESS:
            self.ball_pos.x = self.WALL_THICKNESS + self.BALL_RADIUS
            self.ball_vel.x *= -1
            # Sound: Wall Bounce
        elif self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_pos.x = self.WIDTH - self.WALL_THICKNESS - self.BALL_RADIUS
            self.ball_vel.x *= -1
            # Sound: Wall Bounce
        if self.ball_pos.y - self.BALL_RADIUS <= self.WALL_THICKNESS:
            self.ball_pos.y = self.WALL_THICKNESS + self.BALL_RADIUS
            self.ball_vel.y *= -1
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # Sound: Paddle Hit
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            self.ball_vel.y *= -1
            
            # Add spin based on where it hit the paddle
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2
            self.ball_vel = self.ball_vel.normalize() * self.ball_speed

        # Block collisions
        hit_block = None
        for block_data in self.blocks:
            if ball_rect.colliderect(block_data["rect"]):
                hit_block = block_data
                break
        
        if hit_block:
            # Sound: Block Hit
            self.blocks.remove(hit_block)
            self.score += hit_block["points"]
            self.blocks_destroyed_count += 1
            reward += 0.1 + hit_block["points"] # Hit reward + destroy reward
            
            self._create_particles(hit_block["rect"].center, hit_block["color"])

            # Determine bounce direction
            prev_ball_pos = self.ball_pos - self.ball_vel
            prev_ball_rect = pygame.Rect(prev_ball_pos.x - self.BALL_RADIUS, prev_ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            
            if prev_ball_rect.bottom <= hit_block["rect"].top or prev_ball_rect.top >= hit_block["rect"].bottom:
                self.ball_vel.y *= -1
            else:
                self.ball_vel.x *= -1
            
            # Difficulty scaling
            if self.blocks_destroyed_count % 20 == 0:
                self.ball_speed = min(self.MAX_BALL_SPEED, self.ball_speed + 0.1)
                self.ball_vel = self.ball_vel.normalize() * self.ball_speed


        # Ball out of bounds
        if self.ball_pos.y - self.BALL_RADIUS > self.HEIGHT:
            # Sound: Lose Ball
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifetime": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9  # Damping
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.balls_left <= 0:
            self.win_state = False
            return True
        if not self.blocks:
            self.win_state = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            # Add a slight 3D effect
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in block["color"]), block["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 30.0))))
            color_with_alpha = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

        # Ball
        if self.balls_left > 0:
            x, y = int(self.ball_pos.x), int(self.ball_pos.y)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS + 2, (100, 100, 100, 50))
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.HEIGHT - 35))

        balls_text = self.font.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 10, self.HEIGHT - 35))

        if self.game_over:
            if self.win_state:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            game_over_surf = self.game_over_font.render(msg, True, color)
            game_over_rect = game_over_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(game_over_surf, game_over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "ball_speed": self.ball_speed
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
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'dummy' as needed

    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different screen setup
    pygame.display.set_caption("Block Breaker")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Main game loop
    running = True
    while running:
        action = [0, 0, 0]  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if done:
            # Display final message for a bit before allowing reset
            if keys[pygame.K_r]:
                obs, info = env.reset()
                done = False

        # Render the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(60) # Control the frame rate of the playable version

    env.close()