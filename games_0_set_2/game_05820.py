
# Generated: 2025-08-28T06:12:10.553641
# Source Brief: brief_05820.md
# Brief Index: 5820

        
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
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to move up/down. "
        "Dodge the obstacles and reach the end of all 3 sections before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-styled side-scrolling racer. Navigate through three increasingly "
        "difficult, procedurally generated track sections against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    W, H = 640, 400
    FPS = 30
    NUM_SECTIONS = 3
    SECTION_LENGTH = 3000  # Pixels per section
    TIME_PER_SECTION = 45  # Seconds

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_TRACK = (60, 60, 80)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_OBSTACLE = (50, 150, 255)
    COLOR_PARTICLE = (255, 200, 100)
    COLOR_FINISH_LINE = (255, 255, 255)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (30, 30, 50, 180)

    # Player
    PLAYER_W, PLAYER_H = 25, 15
    PLAYER_ACCEL = 0.5
    PLAYER_BRAKE = 1.0
    PLAYER_MAX_SPEED = 15.0
    PLAYER_FRICTION = 0.97
    PLAYER_VERT_SPEED = 6

    # Track
    TRACK_H_TOP = 100
    TRACK_H_BOTTOM = 300
    TRACK_BORDER_H = 5

    # Obstacles
    OBSTACLE_BASE_DENSITY = 0.005  # Obstacles per pixel
    OBSTACLE_SPEED_MOD = 0.2
    OBSTACLE_DENSITY_MOD = 0.25
    OBSTACLE_MIN_GAP = PLAYER_H * 2.5
    OBSTACLE_W = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # These are initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel_x = 0
        self.camera_x = 0
        self.current_section = 0
        self.time_remaining = 0
        self.obstacles = []
        self.finish_lines = []
        self.particles = []
        self.parallax_stars = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Rect(100, self.H / 2 - self.PLAYER_H / 2, self.PLAYER_W, self.PLAYER_H)
        self.player_vel_x = 0
        self.camera_x = 0
        self.current_section = 0
        self.time_remaining = self.NUM_SECTIONS * self.TIME_PER_SECTION * self.FPS

        self._generate_track()
        self._generate_parallax_stars()

        return self._get_observation(), self._get_info()

    def _generate_track(self):
        self.obstacles = []
        self.finish_lines = []
        for i in range(self.NUM_SECTIONS):
            section_start = i * self.SECTION_LENGTH
            self._generate_obstacles_for_section(i)
            if i < self.NUM_SECTIONS:
                finish_x = section_start + self.SECTION_LENGTH
                self.finish_lines.append(pygame.Rect(finish_x, self.TRACK_H_TOP, 10, self.TRACK_H_BOTTOM - self.TRACK_H_TOP))

    def _generate_obstacles_for_section(self, section_idx):
        track_height = self.TRACK_H_BOTTOM - self.TRACK_H_TOP
        section_start = section_idx * self.SECTION_LENGTH
        section_end = section_start + self.SECTION_LENGTH
        
        density = self.OBSTACLE_BASE_DENSITY * (1 + self.OBSTACLE_DENSITY_MOD * section_idx)
        num_obstacles = int((section_end - section_start) * density)

        for _ in range(num_obstacles):
            x = self.np_random.uniform(section_start + 100, section_end - 100)
            
            # Ensure a gap is always present
            max_obs_height = track_height - self.OBSTACLE_MIN_GAP
            h = self.np_random.uniform(20, max_obs_height)
            
            is_top = self.np_random.choice([True, False])
            y = self.TRACK_H_TOP if is_top else self.TRACK_H_BOTTOM - h
            
            obstacle_rect = pygame.Rect(x, y, self.OBSTACLE_W, h)
            
            is_moving = self.np_random.random() < (0.2 + 0.1 * section_idx)
            if is_moving:
                speed = (self.np_random.random() + 0.5) * (1 + self.OBSTACLE_SPEED_MOD * section_idx)
                direction = self.np_random.choice([-1, 1])
                self.obstacles.append({"rect": obstacle_rect, "moving": True, "speed": speed, "dir": direction})
            else:
                self.obstacles.append({"rect": obstacle_rect, "moving": False})

    def _generate_parallax_stars(self):
        self.parallax_stars = []
        for _ in range(200):
            x = self.np_random.uniform(0, self.W)
            y = self.np_random.uniform(0, self.H)
            depth = self.np_random.uniform(0.1, 0.7) # Slower scrolling for distant stars
            self.parallax_stars.append([x, y, depth])

    def step(self, action):
        if self.game_over:
            # If game is over, just return the final state without updates
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Update Player ---
        if movement == 1:  # Accelerate
            self.player_vel_x += self.PLAYER_ACCEL
        elif movement == 2:  # Brake
            self.player_vel_x -= self.PLAYER_BRAKE
            reward -= 0.2 # Penalty for braking
        if movement == 3:  # Move Up
            self.player_pos.y -= self.PLAYER_VERT_SPEED
        elif movement == 4:  # Move Down
            self.player_pos.y += self.PLAYER_VERT_SPEED

        self.player_vel_x *= self.PLAYER_FRICTION
        self.player_vel_x = max(0, min(self.player_vel_x, self.PLAYER_MAX_SPEED))
        self.player_pos.x += self.player_vel_x
        
        # Clamp player to track
        self.player_pos.y = max(self.TRACK_H_TOP, min(self.player_pos.y, self.TRACK_H_BOTTOM - self.PLAYER_H))
        
        if self.player_vel_x > 1.0:
            reward += 0.1 # Reward for forward movement

        # --- Update Game World ---
        self.camera_x = self.player_pos.x - 100
        self.steps += 1
        self.time_remaining -= 1

        self._update_entities()

        # --- Check for Events ---
        terminated = False
        
        # Collision
        if self._check_collision():
            # sfx: explosion
            self._create_particles(self.player_pos.centerx, self.player_pos.centery, 50)
            reward -= 10
            self.game_over = True
            terminated = True
        
        # Section complete
        if self.current_section < len(self.finish_lines):
            if self.player_pos.right > self.finish_lines[self.current_section].x:
                # sfx: checkpoint
                reward += 10
                self.current_section += 1
                if self.current_section == self.NUM_SECTIONS:
                    # sfx: victory
                    reward += 100
                    self.game_over = True
                    terminated = True

        # Time out
        if self.time_remaining <= 0:
            # sfx: game over
            self.game_over = True
            terminated = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_entities(self):
        # Update moving obstacles
        for obs in self.obstacles:
            if obs["moving"]:
                obs["rect"].y += obs["speed"] * obs["dir"]
                if obs["rect"].top < self.TRACK_H_TOP or obs["rect"].bottom > self.TRACK_H_BOTTOM:
                    obs["dir"] *= -1
                    obs["rect"].y = max(self.TRACK_H_TOP, min(obs["rect"].y, self.TRACK_H_BOTTOM - obs["rect"].height))
        
        # Update particles
        for p in self.particles[:]:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1 # lifetime
            if p[4] <= 0:
                self.particles.remove(p)

    def _check_collision(self):
        for obs in self.obstacles:
            if self.player_pos.colliderect(obs["rect"]):
                return True
        return False

    def _create_particles(self, x, y, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([x, y, vx, vy, lifetime])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_track()
        self._render_entities()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.parallax_stars:
            screen_x = (star[0] - self.camera_x * star[2]) % self.W
            screen_y = star[1]
            intensity = int(100 + 155 * star[2])
            pygame.gfxdraw.pixel(self.screen, int(screen_x), int(screen_y), (intensity, intensity, intensity))

    def _render_track(self):
        # Main track area
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_H_TOP, self.W, self.TRACK_H_BOTTOM - self.TRACK_H_TOP))
        # Top and bottom borders
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, self.TRACK_H_TOP - self.TRACK_BORDER_H, self.W, self.TRACK_BORDER_H))
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, self.TRACK_H_BOTTOM, self.W, self.TRACK_BORDER_H))

    def _render_entities(self):
        # Obstacles
        for obs in self.obstacles:
            screen_rect = obs["rect"].move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.W:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

        # Finish Lines
        for line in self.finish_lines:
            screen_rect = line.move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.W:
                check_h = 10
                for i in range(line.height // check_h):
                    color = self.COLOR_FINISH_LINE if i % 2 == 0 else (0,0,0)
                    pygame.draw.rect(self.screen, color, (screen_rect.x, screen_rect.y + i * check_h, line.width, check_h))

        # Particles
        for p in self.particles:
            screen_x, screen_y = p[0] - self.camera_x, p[1]
            alpha = max(0, min(255, int(255 * (p[4] / 20))))
            color = (self.COLOR_PARTICLE[0], self.COLOR_PARTICLE[1], self.COLOR_PARTICLE[2], alpha)
            
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(screen_x), int(screen_y)))

    def _render_player(self):
        screen_rect = self.player_pos.move(-self.camera_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_rect)
        # Add a small "cockpit" to show direction
        cockpit_rect = pygame.Rect(screen_rect.right - 5, screen_rect.centery - 2, 5, 4)
        pygame.draw.rect(self.screen, (200, 200, 255), cockpit_rect)

    def _render_ui(self):
        # Semi-transparent background for UI
        ui_panel = pygame.Surface((self.W, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        time_str = f"TIME: {max(0, self.time_remaining / self.FPS):.1f}"
        self._draw_text(time_str, (10, 10))

        section_str = f"SECTION: {min(self.current_section + 1, self.NUM_SECTIONS)}/{self.NUM_SECTIONS}"
        self._draw_text(section_str, (self.W - 150, 10))

        score_str = f"SCORE: {self.score:.1f}"
        self._draw_text(score_str, (self.W / 2 - 50, 10))
        
        if self.game_over:
            if self.current_section == self.NUM_SECTIONS:
                msg = "RACE COMPLETE!"
            elif self.time_remaining <= 0:
                msg = "TIME UP!"
            else:
                msg = "CRASHED!"
            
            self._draw_text(msg, (self.W/2, self.H/2 - 30), self.font_large, center=True)


    def _draw_text(self, text, pos, font=None, color=None, center=False):
        if font is None: font = self.font_small
        if color is None: color = self.COLOR_UI_TEXT
        
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "section": self.current_section,
            "time_remaining": self.time_remaining / self.FPS,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.W, GameEnv.H))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}")
            # Keep showing the final screen for a moment before auto-resetting
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Wait 2 seconds

            print("Resetting environment.")
            obs, info = env.reset()
            total_reward = 0

        # --- Display the game screen ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()