
# Generated: 2025-08-27T20:15:27.482176
# Source Brief: brief_02402.md
# Brief Index: 2402

        
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
        "Controls: Press ↑ or SPACE to jump. The game scrolls automatically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-scrolling platformer. Time your jumps to cross the "
        "procedurally generated level and reach the red flag before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.font_s = pygame.font.SysFont("Arial", 18)
        self.font_m = pygame.font.SysFont("Arial", 24)
        self.COLOR_BG_TOP = (40, 60, 100)
        self.COLOR_BG_BOTTOM = (80, 120, 180)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 110, 120)
        self.COLOR_FLAG = (220, 50, 50)
        self.COLOR_FLAGPOLE = (180, 180, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.background = self._create_gradient_background()
        
        # --- Game Constants ---
        self.gravity = 0.5
        self.jump_strength = -10
        self.player_size = 20
        self.max_level = 3
        self.coyote_time_frames = 4
        self.level_time_seconds = 120
        self.max_episode_steps = 10000

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.platforms = None
        self.flag_pos = None
        self.camera_y = None
        self.on_ground = None
        self.coyote_timer = None
        self.particles = None
        self.player_trail = None
        self.camera_shake = None
        
        self.level = 1
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        
        # --- Final Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.level_time_seconds * 30  # 30 FPS
        
        self.player_pos = pygame.Vector2(self.screen_width / 4, self.screen_height / 2)
        self.player_vel = pygame.Vector2(0, 0)
        
        self.on_ground = False
        self.coyote_timer = 0
        
        self.particles = []
        self.player_trail = []
        self.camera_shake = 0
        
        # If all levels are cleared, reset to level 1
        if options and "level" in options:
            self.level = options.get("level", 1)
        elif self.level > self.max_level:
             self.level = 1

        self._generate_platforms()
        self.camera_y = self.player_pos.y - self.screen_height / 2

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms = []
        plat_y = self.screen_height * 0.75
        plat_x = 50
        
        # Starting platform
        start_plat = pygame.Rect(plat_x - 100, plat_y, 300, 20)
        self.platforms.append(start_plat)
        plat_x += start_plat.width

        # Procedurally generate level
        num_platforms = 20
        scroll_speed_mod = 1 + (self.level - 1) * 0.1
        gap_mod = (self.level - 1) * 5

        for i in range(num_platforms):
            gap = self.np_random.integers(100, 150) + gap_mod
            plat_x += gap
            
            width = self.np_random.integers(100, 250)
            height_change = self.np_random.integers(-80, 80)
            plat_y = np.clip(plat_y + height_change, 100, self.screen_height - 50)
            
            new_plat = pygame.Rect(plat_x, plat_y, width, 20)
            self.platforms.append(new_plat)
            plat_x += width

        # Final platform with flag
        final_plat = pygame.Rect(plat_x + 200, self.screen_height * 0.75, 400, 100)
        self.platforms.append(final_plat)
        self.flag_pos = pygame.Vector2(final_plat.x + 50, final_plat.y)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.01  # Small reward for surviving
        
        # --- Handle player input ---
        jump_attempted = (movement == 1 or space_held)
        if jump_attempted and (self.on_ground or self.coyote_timer > 0):
            self.player_vel.y = self.jump_strength
            self.on_ground = False
            self.coyote_timer = 0
            # sfx: jump

        # --- Update game logic ---
        self.steps += 1
        self.timer -= 1

        # Physics
        self.player_vel.y += self.gravity
        self.player_pos += self.player_vel
        
        if self.on_ground:
            self.coyote_timer = self.coyote_time_frames
        else:
            self.coyote_timer = max(0, self.coyote_timer - 1)

        # Scrolling
        scroll_speed = (2.0 + (self.level - 1) * 0.2)
        for plat in self.platforms:
            plat.x -= scroll_speed
        self.flag_pos.x -= scroll_speed

        # Collision detection
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.player_size, self.player_size)
        
        prev_highest_platform_y = self.screen_height
        is_over_platform = False

        for plat in self.platforms:
            # Check if player is horizontally aligned with a platform
            if player_rect.right > plat.left and player_rect.left < plat.right:
                is_over_platform = True
                prev_highest_platform_y = min(prev_highest_platform_y, plat.top)
                
                # Check for landing collision
                if self.player_vel.y >= 0 and player_rect.bottom >= plat.top > player_rect.bottom - self.player_vel.y - 1:
                    self.player_pos.y = plat.top - self.player_size
                    self.player_vel.y = 0
                    self.on_ground = True
                    if not self.player_trail: # First land
                         reward += 1.0
                         self.score += 10
                         self._create_particles(self.player_pos + pygame.Vector2(self.player_size/2, self.player_size), 15)
                         self.camera_shake = 10
                         # sfx: land
                    break
        
        # Near miss penalty
        if not self.on_ground and is_over_platform and player_rect.top > prev_highest_platform_y:
            reward -= 1.0
            self.score -= 1


        # Update trail
        self.player_trail.append(player_rect.copy())
        if len(self.player_trail) > 5:
            self.player_trail.pop(0)
            
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update camera
        self.camera_y = self.camera_y * 0.9 + (self.player_pos.y - self.screen_height / 2) * 0.1
        self.camera_shake = max(0, self.camera_shake - 1)

        # --- Check termination conditions ---
        terminated = False
        # Win condition
        if player_rect.colliderect(pygame.Rect(self.flag_pos.x, self.flag_pos.y - 60, 10, 60)):
            win_bonus = 100 * (self.timer / (self.level_time_seconds * 30))
            reward += win_bonus
            self.score += int(win_bonus * 10)
            self.game_over = True
            terminated = True
            self.level += 1
            # sfx: win

        # Loss conditions
        if self.player_pos.y > self.screen_height:
            reward -= 100
            self.score -= 1000
            self.game_over = True
            terminated = True
            # sfx: fall
        
        if self.timer <= 0:
            reward -= 100
            self.score -= 500
            self.game_over = True
            terminated = True
            # sfx: timeout
            
        if self.steps >= self.max_episode_steps:
             self.game_over = True
             terminated = True

        self.score = max(0, self.score)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.background, (0, 0))
        
        # Apply camera shake
        shake_offset = pygame.Vector2(0, 0)
        if self.camera_shake > 0:
            shake_offset.x = self.np_random.integers(-self.camera_shake, self.camera_shake)
            shake_offset.y = self.np_random.integers(-self.camera_shake, self.camera_shake)

        # Render all game elements
        self._render_game(shake_offset)
        
        # Render UI overlay (not affected by camera)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer / 30, # in seconds
        }

    def _render_game(self, shake_offset):
        # Calculate camera offset
        cam_offset = pygame.Vector2(0, self.camera_y) - shake_offset

        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_offset))

        # Render flag
        if self.flag_pos:
            flag_pole_rect = pygame.Rect(self.flag_pos.x, self.flag_pos.y - 60, 5, 60)
            pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, flag_pole_rect.move(-cam_offset))
            
            wave = math.sin(self.steps * 0.2) * 5
            flag_points = [
                (self.flag_pos.x + 5, self.flag_pos.y - 60),
                (self.flag_pos.x + 45, self.flag_pos.y - 50 + wave),
                (self.flag_pos.x + 5, self.flag_pos.y - 40)
            ]
            flag_points_offset = [(p[0] - cam_offset.x, p[1] - cam_offset.y) for p in flag_points]
            pygame.gfxdraw.aapolygon(self.screen, flag_points_offset, self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, flag_points_offset, self.COLOR_FLAG)


        # Render player trail
        for i, trail_rect in enumerate(self.player_trail):
            alpha = (i + 1) * 30
            trail_surface = pygame.Surface((self.player_size, self.player_size), pygame.SRCALPHA)
            trail_surface.fill((*self.COLOR_PLAYER, alpha))
            self.screen.blit(trail_surface, trail_rect.topleft - cam_offset)

        # Render player
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.player_size, self.player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.move(-cam_offset))

        # Render particles
        for p in self.particles:
            alpha = max(0, p['life'] * 15)
            color = (*p['color'], alpha)
            particle_surface = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surface, color, (p['size'], p['size']), p['size'])
            self.screen.blit(particle_surface, p['pos'] - pygame.Vector2(p['size'], p['size']) - cam_offset)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {int(self.timer / 30):03d}"
        time_surf = self.font_m.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Level
        level_text = f"LEVEL: {self.level}/{self.max_level}"
        level_surf = self.font_m.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (self.screen_width - level_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score:05d}"
        score_surf = self.font_m.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.screen_width / 2 - score_surf.get_width() / 2, 10))

    def _create_gradient_background(self):
        bg = pygame.Surface((self.screen_width, self.screen_height))
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(bg, (r, g, b), (0, y), (self.screen_width, y))
        return bg

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.integers(2, 5),
                'color': self.COLOR_PLAYER,
            })
            
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Minimalist Platformer")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    
    # --- Main Game Loop ---
    while not terminated:
        # --- Action Mapping for Human ---
        # Default action is no-op
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses a different coordinate system for surfaces, so we need to transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()

        if terminated and info.get("level", 1) <= env.max_level:
             print(f"Level complete! Score: {info['score']}. Starting next level...")
             obs, info = env.reset(options={"level": info["level"]})
             terminated = False
        elif terminated:
             print(f"Game Over! Final Score: {info['score']}")

        clock.tick(30) # Match the environment's internal clock

    env.close()
    pygame.quit()