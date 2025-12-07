
# Generated: 2025-08-27T13:52:18.395151
# Source Brief: brief_00510.md
# Brief Index: 510

        
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


# Particle class for explosion effects
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.lifespan = 30  # 1 second at 30fps
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.radius = random.randint(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        # Add slight gravity/drag
        self.vy += 0.1
        self.vx *= 0.98

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, top-down block breaker. Destroy all blocks to win, but lose all 3 balls and you lose."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_MAX_SPEED = 8
        self.MAX_STEPS = 1000 

        # Colors
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (0, 255, 0),    # Green
            (255, 69, 0),   # Red-Orange
        ]

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.score = None
        self.balls_left = None
        self.steps = None
        self.game_over = None
        self.particles = []
        self.combo_timer = 0
        self.combo_count = 0
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect((self.WIDTH - self.PADDLE_WIDTH) // 2, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Initialize blocks
        self.blocks = []
        block_width, block_height = 58, 20
        rows, cols = 5, 10
        for i in range(rows):
            for j in range(cols):
                block_x = j * (block_width + 2) + 20
                block_y = i * (block_height + 2) + 40
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(block_x, block_y, block_width, block_height), "color": color})

        # Initialize other game state
        self.score = 0
        self.balls_left = 3
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.combo_timer = 0
        self.combo_count = 0

        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball = pygame.Rect(self.paddle.centerx - self.BALL_RADIUS, self.paddle.top - self.BALL_RADIUS * 2, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        if self.auto_advance:
            self.clock.tick(30)

        self.steps += 1
        reward = -0.02  # Small penalty for time passing

        # -- ACTION HANDLING --
        movement = action[0]
        space_held = action[1] == 1
        
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # Sound: Ball Launch
            initial_angle = random.uniform(-math.pi * 3/4, -math.pi * 1/4) # Upwards cone
            self.ball_vel = [math.cos(initial_angle) * self.BALL_MAX_SPEED * 0.75, math.sin(initial_angle) * self.BALL_MAX_SPEED * 0.75]

        # -- GAME LOGIC --
        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_count = 0

        # Ball logic
        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

            # Wall collisions
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.clamp_ip(self.screen.get_rect())
                # Sound: Wall Bounce
            if self.ball.top <= 0:
                self.ball_vel[1] *= -1
                self.ball.clamp_ip(self.screen.get_rect())
                # Sound: Wall Bounce

            # Bottom wall collision (lose life)
            if self.ball.top >= self.HEIGHT:
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._reset_ball()
                    # Sound: Lose Life
                else:
                    self.game_over = True
                    reward -= 100
                    # Sound: Game Over

            # Paddle collision
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self.ball.bottom = self.paddle.top
                # Sound: Paddle Hit
                
                # Change angle based on hit position
                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = offset * self.BALL_MAX_SPEED
                self.ball_vel[1] *= -1
                
                # Ensure ball speed is consistent
                speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                if speed > 0 and speed > self.BALL_MAX_SPEED:
                    scale = self.BALL_MAX_SPEED / speed
                    self.ball_vel = [self.ball_vel[0] * scale, self.ball_vel[1] * scale]


            # Block collisions
            hit_block = None
            for block_data in self.blocks:
                if self.ball.colliderect(block_data["rect"]):
                    hit_block = block_data
                    break
            
            if hit_block:
                self.blocks.remove(hit_block)
                self.score += 1
                # Sound: Block Break
                
                # Create particles
                for _ in range(15):
                    self.particles.append(Particle(hit_block["rect"].centerx, hit_block["rect"].centery, hit_block["color"]))

                # Reward
                reward += 1
                if self.combo_timer > 0:
                    self.combo_count += 1
                    reward += 5 # Combo reward
                self.combo_timer = 15 # 0.5s at 30fps

                # Ball reflection logic
                # Determine if hit was more horizontal or vertical
                overlap = self.ball.clip(hit_block["rect"])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
        
        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

        # -- TERMINATION CHECK --
        if not self.blocks: # Win condition
            self.game_over = True
            reward += 100
            self.score += 100
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            # Add a slight 3D effect
            brighter = tuple(min(255, c + 30) for c in block_data["color"])
            darker = tuple(max(0, c - 30) for c in block_data["color"])
            pygame.draw.line(self.screen, brighter, block_data["rect"].topleft, block_data["rect"].topright, 2)
            pygame.draw.line(self.screen, brighter, block_data["rect"].topleft, block_data["rect"].bottomleft, 2)
            pygame.draw.line(self.screen, darker, block_data["rect"].bottomright, block_data["rect"].topright, 2)
            pygame.draw.line(self.screen, darker, block_data["rect"].bottomright, block_data["rect"].bottomleft, 2)

        # Draw ball with anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render balls left
        balls_text = self.font_main.render(f"Balls: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Render game over messages
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # To control with keyboard:
    # pip install keyboard
    try:
        import keyboard
        print("\n" + "="*50)
        print("Human Controls Enabled:")
        print(env.user_guide)
        print("="*50 + "\n")
        human_control = True
    except ImportError:
        print("\n" + "="*50)
        print("Running with random agent.")
        print("Install 'keyboard' (`pip install keyboard`) to play.")
        print("="*50 + "\n")
        human_control = False

    # Main game loop
    render_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    while not done:
        action = [0, 0, 0] # Default no-op
        
        if human_control:
            # Movement
            if keyboard.is_pressed('left arrow') or keyboard.is_pressed('a'):
                action[0] = 3
            elif keyboard.is_pressed('right arrow') or keyboard.is_pressed('d'):
                action[0] = 4
            
            # Space
            if keyboard.is_pressed('space'):
                action[1] = 1

        else: # Random agent
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(render_surface, frame)
        pygame.display.flip()

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        if done:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()