
# Generated: 2025-08-28T06:00:24.997498
# Source Brief: brief_05758.md
# Brief Index: 5758

        
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
        "A fast-paced, retro-arcade block-breaking game. Bounce the ball to destroy all blocks. "
        "Risky shots off the edge of the paddle are rewarded."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 12
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 5.0
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 42, 18
        self.MAX_STEPS = 5000
        self.MAX_BOUNCE_ANGLE = math.pi / 3  # 60 degrees

        # --- Colors ---
        self.COLOR_BG_TOP = (16, 16, 48)
        self.COLOR_BG_BOTTOM = (32, 32, 72)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (200, 200, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 51, 51),   # Red
            (255, 153, 51),  # Orange
            (51, 255, 51),   # Green
            (51, 51, 255),   # Blue
            (153, 51, 255),  # Purple
        ]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
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
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_launched = False
        self._reset_ball()
        self._generate_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1

        reward = 0
        block_broken_this_step = False

        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and not self.ball_launched:
            self.ball_launched = True
            # Sound: Ball Launch
            self.ball_vel = [
                (self.np_random.random() - 0.5) * 2,
                -self.BALL_SPEED * 0.9, # Initial velocity mostly upwards
            ]
            # Normalize velocity
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if norm > 0:
                self.ball_vel = [v / norm * self.BALL_SPEED for v in self.ball_vel]


        # --- Update Game State ---
        self.steps += 1
        
        # Continuous rewards
        if self.ball_launched:
            reward += 0.1 # Reward for keeping ball in play

        # Update ball and check for events
        events = self._update_ball()
        if events["block_broken"]:
            block_broken_this_step = True
            reward += 1.0 # Reward for breaking a block
        if events["risky_shot"]:
            reward += 5.0 # Reward for a risky shot

        # Risk penalty for not breaking a block
        if self.ball_launched and not block_broken_this_step:
            reward -= 0.2

        if events["ball_lost"]:
            self.balls_left -= 1
            self.ball_launched = False
            self._reset_ball()
            if self.balls_left <= 0:
                self.game_over = True
                reward -= 100.0 # Penalty for losing
                # Sound: Game Over

        self._update_particles()
        
        # Check for win condition
        if not self.blocks:
            self.game_over = True
            reward += 100.0 # Reward for clearing the stage
            # Sound: Stage Clear

        # Check for termination by max steps
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        num_rows = 5
        num_cols = 14
        total_block_width = num_cols * self.BLOCK_WIDTH
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                block_x = start_x + j * self.BLOCK_WIDTH
                block_y = start_y + i * self.BLOCK_HEIGHT
                rect = pygame.Rect(block_x, block_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": rect, "color": color})

    def _update_ball(self):
        events = {"block_broken": False, "risky_shot": False, "ball_lost": False}
        if not self.ball_launched:
            self._reset_ball()
            return events

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_vel[0] *= -1
            # Sound: Wall Bounce
        if ball_rect.right >= self.WIDTH:
            ball_rect.right = self.WIDTH
            self.ball_vel[0] *= -1
            # Sound: Wall Bounce
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1
            # Sound: Wall Bounce
        
        # Bottom wall (lose ball)
        if ball_rect.top >= self.HEIGHT:
            events["ball_lost"] = True
            # Sound: Ball Lost
            return events

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: Paddle Hit
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            offset = np.clip(offset, -1, 1)
            
            # Check for risky shot
            if abs(offset) > 0.9:
                events["risky_shot"] = True
                # Sound: Risky Shot
            
            angle = offset * self.MAX_BOUNCE_ANGLE
            
            self.ball_vel[0] = self.BALL_SPEED * math.sin(angle)
            self.ball_vel[1] = -self.BALL_SPEED * math.cos(angle)
            
            # Ensure ball is placed outside paddle to prevent re-collision
            ball_rect.bottom = self.paddle.top

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                events["block_broken"] = True
                self.score += 10
                # Sound: Block Break
                
                # Create particles
                for _ in range(10):
                    particle_vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)]
                    self.particles.append({
                        "pos": list(block["rect"].center),
                        "vel": particle_vel,
                        "lifespan": self.np_random.integers(15, 30),
                        "color": block["color"]
                    })
                
                # Determine bounce direction
                prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel[0], ball_rect.y - self.ball_vel[1], ball_rect.width, ball_rect.height)
                
                # Collided from top/bottom
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel[1] *= -1
                # Collided from sides
                else:
                    self.ball_vel[0] *= -1

                self.blocks.remove(block)
                break
        
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return events

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, block["rect"], 1)

        # Render paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 60), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=6)

        # Render ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            size = int(p["lifespan"] / 10) + 1
            surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(surf, color, surf.get_rect())
            self.screen.blit(surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render remaining balls
        for i in range(self.balls_left - 1): # -1 because current ball is not shown
            ball_icon_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, ball_icon_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball_icon_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            
        if self.game_over:
            message = "GAME OVER" if self.balls_left <= 0 else "STAGE CLEAR!"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]
    
    # Use a separate pygame window for rendering if run directly
    live_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # Reset actions each frame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the live screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        live_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()