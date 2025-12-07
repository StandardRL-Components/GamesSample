import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to aim your jump left or right. "
        "Hold Space for a medium jump, or Shift for a powerful jump. "
        "No key press results in a small hop."
    )

    game_description = (
        "A fast-paced arcade platformer. Hop your space creature from platform to platform "
        "to reach the top. Use bonus platforms for extra points, but be quick! "
        "The level is collapsing below you and time is running out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_PLAYER = (100, 255, 100)
        self.COLOR_PLAYER_GLOW = (150, 255, 150, 50)
        self.COLOR_PLATFORM = (150, 160, 180)
        self.COLOR_PLATFORM_BONUS = (255, 215, 0)
        self.COLOR_PLATFORM_TOP = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_OUTLINE = (20, 20, 20)
        self.PARTICLE_COLORS = [(255, 255, 100), (255, 150, 50)]

        # --- Physics & Gameplay ---
        self.GRAVITY = 0.4
        self.JUMP_POWER_SMALL = -8
        self.JUMP_POWER_MEDIUM = -11
        self.JUMP_POWER_LARGE = -14
        self.JUMP_HORIZONTAL_SPEED = 5
        self.PLAYER_SIZE = (16, 24)
        self.MAX_STAGES = 3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # --- Persistent State ---
        self.current_stage = 1

        # --- Initialize State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.platforms = None
        self.camera_y = None
        self.camera_x = None
        self.stars = None
        self.particles = None
        self.max_height_reached = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.won = None
        self.rng = np.random.default_rng()

        # self.reset() # Not needed here as it's called by wrappers

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # On the very first run, self.won might not be defined.
        if getattr(self, 'won', False) and self.current_stage < self.MAX_STAGES:
            self.current_stage += 1
        elif not getattr(self, 'won', True): # If lost or first run
            self.current_stage = 1

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.won = False
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_vel = [0, 0]
        # Player starts in the air, falling towards the first platform
        self.on_ground = False
        self.player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)

        self._generate_platforms()
        self._generate_stars()
        
        # Set camera based on starting platform, not player's initial air position
        start_plat_y = self.platforms[0]['rect'].centery
        self.camera_y = start_plat_y - self.HEIGHT * 0.75
        self.camera_x = self.player_pos[0] - self.WIDTH / 2
        self.max_height_reached = self.player_pos[1]
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # After termination, return the last state repeatedly
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        # --- Handle Input and Jumping ---
        if self.on_ground:
            # FIX: The game is auto-advancing, so the player should always perform a small hop
            # when on the ground, even for a no-op action. The action modifies the jump.
            self._handle_jump(movement, space_held, shift_held)
            self.on_ground = False

        # --- Update Player Physics ---
        old_y = self.player_pos[1]
        self.player_vel[1] += self.GRAVITY
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Horizontal screen wrapping
        if self.player_pos[0] > self.WIDTH: self.player_pos[0] = 0
        if self.player_pos[0] < 0: self.player_pos[0] = self.WIDTH

        # Update height-based reward
        if self.player_pos[1] < self.max_height_reached:
            reward += 0.1 * (self.max_height_reached - self.player_pos[1])
            self.max_height_reached = self.player_pos[1]
        elif self.player_pos[1] > old_y:
            reward -= 0.02 # Penalty for falling

        # --- Collision Detection ---
        self.player_rect.midbottom = (int(self.player_pos[0]), int(self.player_pos[1]))
        if self.player_vel[1] > 0: # Only check for landing when falling
            # Assume not on ground until a collision proves otherwise
            self.on_ground = False
            for plat in self.platforms:
                # Check for collision and that the player was above the platform last frame
                if self.player_rect.colliderect(plat['rect']) and (self.player_pos[1] - self.player_vel[1]) <= plat['rect'].top + 1:
                    self.player_pos[1] = plat['rect'].top
                    self.player_vel = [0, 0]
                    self.on_ground = True
                    self._create_particles(self.player_rect.midbottom, 10, plat['color'])
                    
                    if plat['type'] == 'bonus' and not plat.get('used', False):
                        reward += 5
                        self.score += 5
                        plat['used'] = True # Mark as used
                    elif plat['type'] == 'top':
                        reward += 100
                        self.score += 100
                        self.won = True
                        self.game_over = True
                    # No penalty for normal platforms, just the time penalty
                    break
        
        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1
        self._update_particles()
        self._update_camera()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos[1] > self.camera_y + self.HEIGHT + 50: # Fell off screen
            reward -= 100
            terminated = True
        if self.timer <= 0: # Time ran out
            reward -= 100
            terminated = True
        if self.won:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_jump(self, movement, space_held, shift_held):
        # Determine jump power
        if shift_held:
            jump_power = self.JUMP_POWER_LARGE
        elif space_held:
            jump_power = self.JUMP_POWER_MEDIUM
        else:
            jump_power = self.JUMP_POWER_SMALL
        
        self.player_vel[1] = jump_power

        # Determine horizontal direction
        if movement == 3: # Left
            self.player_vel[0] = -self.JUMP_HORIZONTAL_SPEED
        elif movement == 4: # Right
            self.player_vel[0] = self.JUMP_HORIZONTAL_SPEED
        else: # Up, Down, or None
            self.player_vel[0] = 0
        
        self._create_particles(self.player_rect.midbottom, 5, self.COLOR_PLAYER)

    def _generate_platforms(self):
        self.platforms = []
        plat_y = self.HEIGHT - 40
        
        # Starting platform
        start_plat = pygame.Rect(self.WIDTH/2 - 50, plat_y, 100, 20)
        self.platforms.append({'rect': start_plat, 'type': 'normal', 'color': self.COLOR_PLATFORM})
        
        # Stage-based difficulty
        gap_variance = 10 + self.current_stage * 10
        bonus_chance = 0.1 + self.current_stage * 0.15

        for i in range(150):
            plat_y -= self.rng.integers(60, 90)
            plat_width = self.rng.integers(60, 100)
            
            last_plat_x = self.platforms[-1]['rect'].centerx
            plat_x = last_plat_x + self.rng.integers(-120 - gap_variance, 120 + gap_variance)
            plat_x = np.clip(plat_x, plat_width / 2, self.WIDTH - plat_width / 2)

            is_bonus = self.rng.random() < bonus_chance
            plat_type = 'bonus' if is_bonus else 'normal'
            plat_color = self.COLOR_PLATFORM_BONUS if is_bonus else self.COLOR_PLATFORM
            
            new_plat = pygame.Rect(0, 0, plat_width, 20)
            new_plat.midtop = (int(plat_x), int(plat_y))
            self.platforms.append({'rect': new_plat, 'type': plat_type, 'color': plat_color})

        # Top platform
        top_plat_y = self.platforms[-1]['rect'].y - 100
        top_plat = pygame.Rect(0, top_plat_y, self.WIDTH, 40)
        self.platforms.append({'rect': top_plat, 'type': 'top', 'color': self.COLOR_PLATFORM_TOP})

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.rng.random() * self.WIDTH
            y = self.rng.random() * self.HEIGHT * 5 - self.HEIGHT * 4 # spread over larger area
            size = self.rng.random() * 1.5
            self.stars.append((x, y, size))

    def _update_camera(self):
        # Smoothly follow player horizontally
        target_cam_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x += (target_cam_x - self.camera_x) * 0.1

        # Follow player up, but not down
        target_cam_y = self.player_pos[1] - self.HEIGHT * 0.6
        if target_cam_y < self.camera_y:
            self.camera_y += (target_cam_y - self.camera_y) * 0.1
        
        # Add a slow "collapse" effect, making the floor rise
        self.camera_y += 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            # Parallax scrolling for stars
            scroll_x = (x - self.camera_x * 0.1) % self.WIDTH
            scroll_y = (y - self.camera_y * 0.1) % self.HEIGHT
            if 0 <= scroll_y <= self.HEIGHT:
                brightness = 150 + 105 * math.sin(self.steps * 0.1 + x) # Twinkle
                color = (min(255, brightness), min(255, brightness), min(255, brightness))
                pygame.draw.circle(self.screen, color, (int(scroll_x), int(scroll_y)), max(0, size))

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            cam_rect = plat['rect'].move(-self.camera_x, -self.camera_y)
            if cam_rect.colliderect(self.screen.get_rect()):
                color = plat['color']
                if plat.get('used', False): # Fade out used bonus platforms
                     color = tuple(c * 0.5 for c in plat['color'])
                pygame.draw.rect(self.screen, color, cam_rect, border_radius=4)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), cam_rect, width=2, border_radius=4)

        # Draw player
        cam_x = int(self.player_pos[0] - self.camera_x)
        cam_y = int(self.player_pos[1] - self.camera_y)
        
        # Dynamic player shape for animation
        if self.on_ground:
            squish = 1 + abs(math.sin(self.steps * 0.2)) * 0.2
            w = self.PLAYER_SIZE[0] * squish
            h = self.PLAYER_SIZE[1] / squish
        else: # In air
            stretch = 1 + min(max(-0.5, self.player_vel[1] * -0.05), 0.5)
            w = self.PLAYER_SIZE[0] / stretch
            h = self.PLAYER_SIZE[1] * stretch

        self.player_rect.size = (int(w), int(h))
        self.player_rect.midbottom = (cam_x, cam_y)

        # Draw glow
        glow_radius = int(max(w, h) * 0.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (self.player_rect.centerx - glow_radius, self.player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=6)

        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1] - self.camera_y))
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['life']))

    def _render_ui(self):
        # Helper to draw outlined text
        def draw_text(text, font, color, pos):
            x, y = pos
            text_surface = font.render(text, True, color)
            outline_surface = font.render(text, True, self.COLOR_TEXT_OUTLINE)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                self.screen.blit(outline_surface, (x + dx, y + dy))
            self.screen.blit(text_surface, pos)

        # Score, Timer, Height
        draw_text(f"SCORE: {int(self.score)}", self.font_ui, self.COLOR_TEXT, (10, 10))
        time_left = max(0, self.timer / self.FPS)
        draw_text(f"TIME: {time_left:.1f}", self.font_ui, self.COLOR_TEXT, (self.WIDTH - 150, 10))
        height = int((self.platforms[0]['rect'].top - self.player_pos[1]) / 10)
        draw_text(f"HEIGHT: {max(0, height)}m", self.font_ui, self.COLOR_TEXT, (10, self.HEIGHT - 30))
        draw_text(f"STAGE: {self.current_stage}/{self.MAX_STAGES}", self.font_ui, self.COLOR_TEXT, (self.WIDTH - 150, self.HEIGHT - 30))

        if self.game_over:
            msg = "STAGE CLEAR!" if self.won else "GAME OVER"
            color = self.COLOR_PLAYER if self.won else self.COLOR_PLATFORM_TOP
            text_surf = self.font_big.render(msg, True, color)
            pos = (self.WIDTH/2 - text_surf.get_width()/2, self.HEIGHT/2 - text_surf.get_height()/2)
            draw_text(msg, self.font_big, color, pos)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "timer": self.timer,
        }
        
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.random() * 3 + 2,
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.1

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # To use, you might need to `pip install pygame`
    # and remove/comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    
    # For headed gameplay, comment out the os.environ line at the top of the file.
    is_headless = "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy"
    
    render_mode = "rgb_array"
    if not is_headless:
        pygame.display.init()
        pygame.font.init()

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset(seed=42)
    
    screen = None
    if not is_headless:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Hopping Space Creature")
    
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Playing Game ---")
    print(env.user_guide)

    while running:
        # --- Player Input ---
        action = [0, 0, 0] # Default no-op action
        if not is_headless:
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            # K_DOWN (2) is unused for jump direction
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_reward = 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        if not is_headless:
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
            if not is_headless:
                pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()