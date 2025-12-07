
# Generated: 2025-08-28T04:25:38.256053
# Source Brief: brief_02315.md
# Brief Index: 2315

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = 30
        self.height = 40
        self.is_grounded = False
        self.run_anim_timer = 0
        self.color = (50, 150, 255)

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def jump(self, jump_strength):
        if self.is_grounded:
            self.vy = -jump_strength
            self.is_grounded = False
            # Sound placeholder: # sfx_jump()

    def update(self, gravity, ground_y):
        # Horizontal movement
        self.x += self.vx

        # Vertical movement (physics)
        self.vy += gravity
        self.y += self.vy

        # Ground collision
        if self.y + self.height >= ground_y:
            self.y = ground_y - self.height
            self.vy = 0
            self.is_grounded = True
        else:
            self.is_grounded = False
        
        # Animation timer
        self.run_anim_timer = (self.run_anim_timer + abs(self.vx) * 0.2) % (2 * math.pi)

    def draw(self, surface, camera_x):
        screen_x = int(self.x - camera_x)
        
        # Simple bobbing animation for running
        bob = math.sin(self.run_anim_timer) * 2 if self.is_grounded and self.vx != 0 else 0
        
        body_rect = pygame.Rect(screen_x, int(self.y + bob), self.width, self.height)
        
        # Draw body with a border
        pygame.draw.rect(surface, self.color, body_rect, border_radius=4)
        pygame.draw.rect(surface, (200, 220, 255), body_rect, width=3, border_radius=4)

        # Draw eye
        eye_x = screen_x + (self.width * 0.7)
        eye_y = int(self.y + bob + self.height * 0.3)
        pygame.draw.circle(surface, (255, 255, 255), (eye_x, eye_y), 5)
        pygame.draw.circle(surface, (0, 0, 0), (eye_x, eye_y), 2)


class Obstacle:
    def __init__(self, x, y, width, height, speed, move_range):
        self.x = x
        self.y = y
        self.base_y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.move_range = move_range
        self.angle = random.uniform(0, 2 * math.pi)
        self.cleared = False
        self.color = (255, 80, 80)

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def update(self):
        # Sinusoidal vertical movement
        self.angle += self.speed * 0.1
        self.y = self.base_y + math.sin(self.angle) * self.move_range

    def draw(self, surface, camera_x):
        screen_x = int(self.x - camera_x)
        obs_rect = pygame.Rect(screen_x, int(self.y), self.width, self.height)
        pygame.draw.rect(surface, self.color, obs_rect, border_radius=3)
        pygame.draw.rect(surface, (255, 180, 180), obs_rect, width=3, border_radius=3)


class Particle:
    def __init__(self, x, y, vx, vy, size, life, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.life = life
        self.max_life = life
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity on particles
        self.life -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface, camera_x):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            screen_x = int(self.x - camera_x)
            screen_y = int(self.y)
            # Using gfxdraw for anti-aliased circles
            pygame.gfxdraw.filled_circle(surface, screen_x, screen_y, int(self.size), color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ←→ to run, ↑ or SPACE to jump. Avoid red obstacles and reach the green finish line."
    game_description = "A fast-paced, side-scrolling platformer. Guide your robot through a hazardous, procedurally generated course against the clock."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_GROUND = (40, 40, 60)
        self.COLOR_FINISH = (80, 255, 80)

        # Game constants
        self.GROUND_Y = self.height - 50
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = 12
        self.MOVE_SPEED = 5.0
        self.MAX_STAGES = 3
        self.STAGE_LENGTH = 3000

        # State variables (initialized in reset)
        self.robot = None
        self.obstacles = []
        self.particles = []
        self.camera_x = 0
        self.stage = 1
        self.stage_timer = 0
        self.finish_line_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stage = 1
        self.score = 0
        self.game_over = False
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.robot = Robot(x=100, y=self.GROUND_Y - 40)
        self.obstacles = []
        self.particles = []
        self.camera_x = 0
        self.finish_line_x = self.STAGE_LENGTH
        
        self.steps = 0
        self.stage_timer = 20 * 30 # 20 seconds at 30fps

        # Procedural obstacle generation
        difficulty = 1.0 + 0.1 * (self.stage - 1)
        obs_speed = 1.0 * difficulty
        
        current_x = 800
        while current_x < self.finish_line_x - 500:
            spacing = self.np_random.integers(250, 400) / difficulty
            current_x += spacing
            
            is_high_obstacle = self.np_random.random() < 0.3
            if is_high_obstacle:
                # High obstacle to slide under (conceptually)
                height = self.np_random.integers(100, 150)
                y = self.GROUND_Y - height - 80 # 80px gap for robot
                move_range = self.np_random.integers(10, 30)
            else:
                # Low obstacle to jump over
                height = self.np_random.integers(40, 80)
                y = self.GROUND_Y - height
                move_range = self.np_random.integers(20, 50)

            width = self.np_random.integers(40, 60)
            self.obstacles.append(Obstacle(current_x, y, width, height, obs_speed, move_range))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty
        
        # 1. Handle Input
        if movement == 3:  # Left
            self.robot.vx = -self.MOVE_SPEED
        elif movement == 4:  # Right
            self.robot.vx = self.MOVE_SPEED
            reward += 0.1
        else:
            self.robot.vx = 0
        
        if movement == 1 or space_held: # Up or Space to jump
            if self.robot.is_grounded:
                self.robot.jump(self.JUMP_STRENGTH)
                # Create jump particles
                for _ in range(5):
                    self.particles.append(Particle(
                        self.robot.rect.centerx, self.robot.rect.bottom,
                        self.np_random.uniform(-1, 1), self.np_random.uniform(0, 2),
                        self.np_random.uniform(2, 5), 20, (200, 200, 200)
                    ))

        # 2. Update Game State
        self.steps += 1
        self.stage_timer -= 1
        
        self.robot.update(self.GRAVITY, self.GROUND_Y)
        self.robot.x = max(self.camera_x + 10, self.robot.x) # Prevent moving off-screen left

        for obs in self.obstacles:
            obs.update()

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        self.camera_x = self.robot.x - self.width / 4

        # 3. Check for Events and Calculate Rewards
        terminated = False
        
        # Obstacle collision
        for obs in self.obstacles:
            if self.robot.rect.colliderect(obs.rect):
                terminated = True
                self.game_over = True
                # Sound placeholder: # sfx_explosion()
                for _ in range(50):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 8)
                    self.particles.append(Particle(
                        self.robot.rect.centerx, self.robot.rect.centery,
                        math.cos(angle) * speed, math.sin(angle) * speed,
                        self.np_random.uniform(3, 8), 60, obs.color
                    ))
                break
        
        # Obstacle clearing reward
        for obs in self.obstacles:
            if not obs.cleared and self.robot.x > obs.x + obs.width:
                obs.cleared = True
                is_slide_under = obs.y + obs.height < self.GROUND_Y - self.robot.height - 10
                if is_slide_under:
                    reward -= 0.2 # Penalty for "sliding" under
                else:
                    reward += 1.0 # Reward for jumping over
                self.score += 1

        # Finish line
        if self.robot.x > self.finish_line_x:
            self.score += 10
            reward += 10
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self._setup_stage()
                # Sound placeholder: # sfx_stage_clear()
            else:
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
                # Sound placeholder: # sfx_game_win()
        
        # Time limit
        if self.stage_timer <= 0:
            terminated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Parallax background grid
        for i in range(0, self.width + 100, 50):
            x = i - int(self.camera_x * 0.5) % 50
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for i in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.width, i))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.width, self.height - self.GROUND_Y))

        # Finish line
        finish_screen_x = self.finish_line_x - self.camera_x
        if finish_screen_x < self.width + 50:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.GROUND_Y), 5)

        # Obstacles
        for obs in self.obstacles:
            if obs.x - self.camera_x < self.width and obs.x + obs.width - self.camera_x > 0:
                obs.draw(self.screen, self.camera_x)

        # Particles
        for p in self.particles:
            p.draw(self.screen, self.camera_x)
        
        # Robot
        if not (self.game_over and any(p.color == (255, 80, 80) for p in self.particles)):
             self.robot.draw(self.screen, self.camera_x)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.stage_timer // 30):02d}"
        time_surf = self.font_ui.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, (255, 255, 255))
        self.screen.blit(score_surf, (self.width // 2 - score_surf.get_width() // 2, 10))
        
        # Stage
        stage_text = f"STAGE: {self.stage}/{self.MAX_STAGES}"
        stage_surf = self.font_ui.render(stage_text, True, (255, 255, 255))
        self.screen.blit(stage_surf, (self.width - stage_surf.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            if self.robot.x > self.finish_line_x:
                msg = "YOU WIN!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = (255, 80, 80)
            
            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.width // 2 - msg_surf.get_width() // 2, self.height // 2 - msg_surf.get_height() // 2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": max(0, self.stage_timer // 30)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display for manual testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # 'windows' for Windows, 'x11' for Linux, 'quartz' for macOS

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.init()
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, released, released

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()