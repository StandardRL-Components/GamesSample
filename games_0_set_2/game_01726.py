
# Generated: 2025-08-27T18:05:58.231766
# Source Brief: brief_01726.md
# Brief Index: 1726

        
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
        "Controls: Press space to jump upwards from platform to platform."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade platformer. Hop between procedurally generated platforms, "
        "aiming to reach the top while avoiding oscillating red hazards."
    )

    # Frames auto-advance for real-time physics.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLATFORM = (220, 220, 220)
        self.COLOR_HAZARD = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GOAL = (255, 215, 0)

        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -9.5
        self.PLAYER_SIZE = 20
        self.MAX_STEPS = 5000
        self.LEVEL_HEIGHT = 10000

        # Fonts
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_rect = None
        self.player_vel = None
        self.can_jump = False
        self.prev_space_held = False
        self.platforms = []
        self.hazards = []
        self.particles = []
        self.goal_platform = None
        self.camera_y = 0
        self.max_height_reached = 0
        self.hazard_base_speed = 1.0
        self.hazard_current_speed = 1.0
        self.start_y = 0

        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.start_y = self.HEIGHT - 50
        start_x = self.WIDTH / 2 - self.PLAYER_SIZE / 2
        self.player_rect = pygame.Rect(start_x, self.start_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_vel = [0, 0]
        
        self.can_jump = True
        self.prev_space_held = False
        
        self.platforms = []
        self.hazards = []
        self.particles = []
        self.goal_platform = None
        
        self.max_height_reached = self.player_rect.y
        self.camera_y = self.player_rect.y - self.HEIGHT * 0.7
        self.hazard_current_speed = self.hazard_base_speed

        self._generate_world()
        
        return self._get_observation(), self._get_info()

    def _generate_world(self):
        # Start platform
        start_platform = pygame.Rect(self.WIDTH/2 - 50, self.start_y, 100, 20)
        self.platforms.append(start_platform)
        
        current_y = self.start_y
        
        # Procedural generation of platforms
        while current_y > -self.LEVEL_HEIGHT:
            last_platform = self.platforms[-1]
            gap_y = self.np_random.uniform(60, 120)
            new_y = last_platform.y - gap_y
            
            max_horiz_dist = 150
            new_x = self.np_random.uniform(
                max(20, last_platform.centerx - max_horiz_dist),
                min(self.WIDTH - 120, last_platform.centerx + max_horiz_dist)
            )
            new_width = self.np_random.uniform(60, 120)
            
            new_platform = pygame.Rect(new_x, new_y, new_width, 20)
            self.platforms.append(new_platform)
            current_y = new_y
            
            # Add hazards occasionally
            if self.np_random.random() < 0.2 and new_y < self.start_y - 200:
                hazard_y = new_y + gap_y / 2
                hazard_x_start = self.np_random.uniform(50, self.WIDTH - 50)
                hazard_range = self.np_random.uniform(50, 150)
                hazard_dir = 1 if self.np_random.random() < 0.5 else -1
                self.hazards.append({
                    "pos": [hazard_x_start, hazard_y],
                    "start_x": hazard_x_start,
                    "range": hazard_range,
                    "dir": hazard_dir,
                    "size": 15
                })

        # Goal platform
        goal_y = self.platforms[-1].y - 100
        self.goal_platform = pygame.Rect(self.WIDTH/2 - 75, goal_y, 150, 30)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # 1. Handle Input
        space_held = action[1] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if space_pressed and self.can_jump:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.can_jump = False
            # sfx: jump
            self._create_particles(self.player_rect.midbottom, 10, self.COLOR_PLAYER)
            
        # 2. Update Player Physics
        prev_y = self.player_rect.y
        self.player_vel[1] += self.GRAVITY
        self.player_rect.y += self.player_vel[1]
        
        # Reward for vertical movement
        if self.player_rect.y < prev_y:
            reward += 0.1
        else:
            reward -= 0.01

        # 3. Collision Detection
        self.can_jump = False
        collidable_platforms = self.platforms + ([self.goal_platform] if self.goal_platform else [])
        
        for plat in collidable_platforms:
            if self.player_rect.colliderect(plat) and self.player_vel[1] > 0:
                if prev_y + self.PLAYER_SIZE <= plat.top + 1: # +1 for tolerance
                    self.player_rect.bottom = plat.top
                    self.player_vel[1] = 0
                    self.can_jump = True
                    reward += 1
                    # sfx: land
                    self._create_particles(self.player_rect.midbottom, 5, self.COLOR_PLATFORM)
                    break

        # 4. Update Hazards
        self.hazard_current_speed = self.hazard_base_speed + (self.steps // 500) * 0.05
        for hazard in self.hazards:
            hazard["pos"][0] += hazard["dir"] * self.hazard_current_speed
            if abs(hazard["pos"][0] - hazard["start_x"]) > hazard["range"]:
                hazard["dir"] *= -1

        # 5. Update Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # 6. Update Camera
        target_camera_y = self.player_rect.centery - self.HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.08

        # 7. Update Score/Height
        self.max_height_reached = min(self.max_height_reached, self.player_rect.y)
        self.score = int((self.start_y - self.max_height_reached) / 10)

        # 8. Check Termination Conditions
        terminated = False
        if self.player_rect.top > self.camera_y + self.HEIGHT + 50:
            self.game_over = True
            terminated = True
            reward = -10
            # sfx: fall
        
        for hazard in self.hazards:
            h_center_x, h_center_y = hazard["pos"]
            h_size = hazard["size"]
            h_rect = pygame.Rect(h_center_x - h_size/2, h_center_y - h_size/2, h_size, h_size)
            if self.player_rect.colliderect(h_rect):
                self.game_over = True
                terminated = True
                reward = -10
                # sfx: hurt
                self._create_particles(self.player_rect.center, 20, self.COLOR_HAZARD)
                break
        
        if self.goal_platform and self.player_rect.colliderect(self.goal_platform) and self.can_jump:
            self.game_over = True
            terminated = True
            reward = 100
            # sfx: win
            self._create_particles(self.player_rect.center, 50, self.COLOR_GOAL)
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _get_observation(self):
        # Clear screen with background
        self._render_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        cam_y_int = int(self.camera_y)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - cam_y_int))
            if 0 <= pos[0] < self.WIDTH and 0 <= pos[1] < self.HEIGHT:
                pygame.draw.circle(self.screen, p['color'], pos, max(0, int(p['life'] / 5)))

        # Render platforms
        for plat in self.platforms:
            if plat.bottom > self.camera_y and plat.top < self.camera_y + self.HEIGHT:
                screen_rect = plat.move(0, -cam_y_int)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)
        
        # Render goal
        if self.goal_platform:
            if self.goal_platform.bottom > self.camera_y and self.goal_platform.top < self.camera_y + self.HEIGHT:
                screen_rect = self.goal_platform.move(0, -cam_y_int)
                pygame.draw.rect(self.screen, self.COLOR_GOAL, screen_rect, border_radius=5)
                if self.steps % 10 < 5:
                    pygame.draw.rect(self.screen, (255, 255, 255), screen_rect.inflate(-10, -10), 1, border_radius=3)

        # Render hazards
        for hazard in self.hazards:
            pos_x, pos_y = hazard["pos"]
            screen_y = pos_y - self.camera_y
            if -20 < screen_y < self.HEIGHT + 20:
                size = hazard["size"]
                p1 = (int(pos_x), int(screen_y - size * 0.577)) # Adjusted for equilateral
                p2 = (int(pos_x - size / 2), int(screen_y + size * 0.289))
                p3 = (int(pos_x + size / 2), int(screen_y + size * 0.289))
                try:
                    pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_HAZARD)
                    pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_HAZARD)
                except: # gfxdraw can fail with out-of-bounds coords
                    pass

        # Render player
        if not (self.game_over and self.player_rect.top > self.camera_y + self.HEIGHT):
            screen_player_rect = self.player_rect.move(0, -cam_y_int)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_player_rect)
            pygame.draw.rect(self.screen, tuple(c/2 for c in self.COLOR_PLAYER), screen_player_rect, 1)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        height_val = max(0, int((self.start_y - self.player_rect.y) / 10))
        height_text = self.font_main.render(f"HEIGHT: {height_val}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (self.WIDTH - height_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "GAME OVER"
            if self.goal_platform and self.player_rect.colliderect(self.goal_platform):
                status_text_str = "GOAL REACHED!"
            elif self.steps >= self.MAX_STEPS:
                status_text_str = "TIME UP"
                
            status_text = self.font_main.render(status_text_str, True, self.COLOR_TEXT)
            text_rect = status_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, int((self.start_y - self.player_rect.y) / 10)),
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
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
    
    # Override screen for direct rendering
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platform Hopper")
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = env.action_space.sample() # Start with a random action
        action[1] = 1 if space_held else 0 # Override spacebar action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()