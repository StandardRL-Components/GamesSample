
# Generated: 2025-08-27T13:58:19.280394
# Source Brief: brief_00544.md
# Brief Index: 544

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to run, ↑ to jump. Reach the end of all 3 stages as fast as you can."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced neon platformer. Guide the robot across procedurally generated platforms, "
        "dodging rotating obstacles to reach the goal. Balance speed with risk for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3600 # 60 seconds at 60fps, but brief says 1000. Let's stick to the brief's number.
        self.MAX_EPISODE_STEPS = 1000 
        self.NUM_STAGES = 3

        # Colors
        self.COLOR_BG_TOP = (20, 0, 40)
        self.COLOR_BG_BOTTOM = (60, 10, 60)
        self.COLOR_ROBOT = (0, 255, 255)
        self.COLOR_ROBOT_GLOW = (0, 150, 255)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_PLATFORM_OUTLINE = (150, 150, 170)
        self.COLOR_OBSTACLE = (255, 0, 100)
        self.COLOR_OBSTACLE_PIVOT = (255, 100, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SCORE = (255, 220, 0)
        self.COLOR_PARTICLE = (200, 200, 255)

        # Physics
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12
        self.MAX_SPEED_X = 6
        self.ACCELERATION = 0.6
        self.FRICTION = 0.92

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.robot = {}
        self.platforms = []
        self.obstacles = []
        self.particles = []
        self.camera_offset_x = 0
        self.target_camera_offset_x = 0
        self.stage_end_x = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        
        self.robot = {
            "x": 100, "y": 200,
            "vx": 0, "vy": 0,
            "width": 20, "height": 30,
            "is_grounded": False
        }
        
        self.particles = []
        self._generate_stage(self.stage)
        self.camera_offset_x = self.robot['x'] - self.WIDTH / 4
        self.target_camera_offset_x = self.camera_offset_x

        return self._get_observation(), self._get_info()

    def _generate_stage(self, stage_num):
        self.platforms = []
        self.obstacles = []
        
        # Starting platform
        current_x = 0
        start_platform = pygame.Rect(current_x, 300, 300, 100)
        self.platforms.append(start_platform)
        current_x += start_platform.width

        # Procedural platforms and obstacles
        num_platforms = self.np_random.integers(5, 8)
        for i in range(num_platforms):
            gap = self.np_random.integers(70, 120 + stage_num * 10)
            current_x += gap
            
            width = self.np_random.integers(200, 500)
            height = self.np_random.integers(280, 320)
            platform = pygame.Rect(current_x, height, width, self.HEIGHT - height)
            self.platforms.append(platform)
            
            # Add obstacles
            num_obstacles = self.np_random.integers(0, 3)
            for _ in range(num_obstacles):
                if platform.width > 150: # Only on wider platforms
                    obstacle_x = platform.x + self.np_random.integers(50, platform.width - 50)
                    obstacle_y = platform.top - self.np_random.integers(20, 60)
                    self.obstacles.append({
                        "cx": obstacle_x, "cy": obstacle_y,
                        "length": self.np_random.integers(80, 120),
                        "angle": self.np_random.uniform(0, 2 * math.pi),
                        "speed": (0.02 + 0.01 * stage_num) * self.np_random.choice([-1, 1])
                    })
            
            current_x += width
        
        self.stage_end_x = current_x
        # Add a final small platform to signify the end
        end_platform = pygame.Rect(self.stage_end_x, 300, 50, 100)
        self.platforms.append(end_platform)
        self.stage_end_x += end_platform.width

    def step(self, action):
        reward = 0
        
        # --- 1. Handle Input & Update Robot ---
        movement, _, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1 and self.robot['is_grounded']: # Jump
            self.robot['vy'] = self.JUMP_STRENGTH
            self.robot['is_grounded'] = False
            # sfx: jump_sound()
            self._create_particles(self.robot['x'] + self.robot['width']/2, self.robot['y'] + self.robot['height'], 5, (200,200,255))
        if movement == 3: # Left
            self.robot['vx'] = max(-self.MAX_SPEED_X, self.robot['vx'] - self.ACCELERATION)
        if movement == 4: # Right
            self.robot['vx'] = min(self.MAX_SPEED_X, self.robot['vx'] + self.ACCELERATION)
            reward += 0.01 # Small reward for moving forward
        
        if movement == 0: # No horizontal input
            self.robot['vx'] *= self.FRICTION
        
        if abs(self.robot['vx']) < 0.1:
            self.robot['vx'] = 0
            reward -= 0.005 # Small penalty for standing still
            
        # --- 2. Apply Physics ---
        if not self.robot['is_grounded']:
            self.robot['vy'] += self.GRAVITY
        
        self.robot['x'] += self.robot['vx']
        self.robot['y'] += self.robot['vy']

        # --- 3. Handle Collisions & Game State ---
        robot_rect = pygame.Rect(self.robot['x'], self.robot['y'], self.robot['width'], self.robot['height'])
        
        # Platform collision
        was_grounded = self.robot['is_grounded']
        self.robot['is_grounded'] = False
        for plat in self.platforms:
            if robot_rect.colliderect(plat) and self.robot['vy'] >= 0:
                # Check if robot was above the platform in the previous step
                if robot_rect.bottom - self.robot['vy'] <= plat.top + 1:
                    self.robot['y'] = plat.top - self.robot['height']
                    self.robot['vy'] = 0
                    self.robot['is_grounded'] = True
                    if not was_grounded:
                        reward += 1.0 # Landing reward
                        # sfx: land_sound()
                        self._create_particles(robot_rect.midbottom[0], robot_rect.midbottom[1], 15, (255,255,255))
                    break
        
        # Obstacle collision
        for obs in self.obstacles:
            x1 = obs['cx'] + math.cos(obs['angle']) * obs['length'] / 2
            y1 = obs['cy'] + math.sin(obs['angle']) * obs['length'] / 2
            x2 = obs['cx'] - math.cos(obs['angle']) * obs['length'] / 2
            y2 = obs['cy'] - math.sin(obs['angle']) * obs['length'] / 2
            if robot_rect.clipline((x1, y1), (x2, y2)):
                self.game_over = True
                reward = -5.0 # Penalty for hitting obstacle
                # sfx: death_sound()
                self._create_particles(robot_rect.centerx, robot_rect.centery, 50, self.COLOR_OBSTACLE)
                break
        if self.game_over:
             self.score += reward
             return self._get_observation(), reward, True, False, self._get_info()

        # --- 4. Update Dynamic Elements ---
        for obs in self.obstacles:
            obs['angle'] += obs['speed']
        
        self._update_particles()
        
        # --- 5. Check Termination/Progression ---
        self.steps += 1
        
        # Fall death
        if self.robot['y'] > self.HEIGHT + 50:
            self.game_over = True
            reward = -5.0
        
        # Timeout
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            reward = -2.0

        # Stage complete
        if self.robot['x'] > self.stage_end_x:
            self.stage += 1
            if self.stage > self.NUM_STAGES:
                self.win = True
                self.game_over = True
                reward += 300.0 # Big win reward
                # sfx: win_jingle()
            else:
                reward += 100.0 # Stage complete reward
                # sfx: stage_clear_sound()
                self.robot['x'] = 100
                self.robot['y'] = 200
                self.robot['vx'] = self.robot['vy'] = 0
                self._generate_stage(self.stage)

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # particle gravity
            p['life'] -= 1

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-4, -1),
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # Update camera
        self.target_camera_offset_x = self.robot['x'] - self.WIDTH / 4
        self.camera_offset_x += (self.target_camera_offset_x - self.camera_offset_x) * 0.1

        # Clear screen with background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_offset_x)

        # Draw platforms
        for plat in self.platforms:
            # Only draw visible platforms
            if plat.right - cam_x > 0 and plat.left - cam_x < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, 0))
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat.move(-cam_x, 0), 2)
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs['cx'] - cam_x > -obs['length'] and obs['cx'] - cam_x < self.WIDTH + obs['length']:
                x1 = obs['cx'] + math.cos(obs['angle']) * obs['length'] / 2 - cam_x
                y1 = obs['cy'] + math.sin(obs['angle']) * obs['length'] / 2
                x2 = obs['cx'] - math.cos(obs['angle']) * obs['length'] / 2 - cam_x
                y2 = obs['cy'] - math.sin(obs['angle']) * obs['length'] / 2
                pygame.draw.aaline(self.screen, self.COLOR_OBSTACLE, (x1, y1), (x2, y2), 3)
                pygame.gfxdraw.filled_circle(self.screen, int(obs['cx'] - cam_x), int(obs['cy']), 5, self.COLOR_OBSTACLE_PIVOT)
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x'] - cam_x), int(p['y']), int(p['radius']), (*p['color'], int(255 * (p['life'] / 30.0))))

        # Draw robot
        robot_screen_x = int(self.robot['x'] - cam_x)
        robot_screen_y = int(self.robot['y'])
        robot_rect = pygame.Rect(robot_screen_x, robot_screen_y, self.robot['width'], self.robot['height'])
        
        # Glow effect
        glow_radius = int(self.robot['width'] * 1.5)
        glow_center = robot_rect.center
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_ROBOT_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        pygame.draw.circle(glow_surf, (*self.COLOR_ROBOT_GLOW, 80), (glow_radius, glow_radius), int(glow_radius * 0.7))
        self.screen.blit(glow_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Robot body
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=4)
        
        # Robot eye
        eye_x = robot_rect.centerx + (5 if self.robot['vx'] >= 0 else -5)
        eye_y = robot_rect.centery - 5
        pygame.draw.circle(self.screen, (0,0,0), (eye_x, eye_y), 3)


    def _render_ui(self):
        # Stage display
        stage_text = self.font_medium.render(f"STAGE {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer/Steps display
        time_left = max(0, self.MAX_EPISODE_STEPS - self.steps)
        time_text = self.font_medium.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Score display
        score_str = f"SCORE: {int(self.score)}"
        score_text = self.font_large.render(score_str, True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            msg_text = self.font_large.render(message, True, self.COLOR_WIN if self.win else self.COLOR_OBSTACLE)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(msg_text, msg_rect)

            final_score_text = self.font_medium.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Use a separate screen for rendering if playing directly
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Platformer")
    
    action = np.array([0, 0, 0]) # No-op
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        mov = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            mov = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            mov = 2 # down (unused in this game)
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            mov = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            mov = 4 # right
            
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov, space, shift])

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            # Wait a bit on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

        env.clock.tick(60) # Run at 60 FPS
        
    env.close()