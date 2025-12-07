import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:05:42.469772
# Source Brief: brief_00810.md
# Brief Index: 810
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a steampunk puzzle-platformer, "Clockwork City".

    The player navigates an intricate clockwork world by flipping gravity and
    building temporary platforms. The goal is to reach the exit portal in each
    level before time runs out.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) relative to gravity.
    - actions[1]: Gravity Flip (0=released, 1=pressed). Triggers on press.
    - actions[2]: Build Platform (0=released, 1=pressed). Triggers on press.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game screen.

    Reward Structure:
    - +100 for completing a level.
    - -100 for falling off-screen.
    - -50 for running out of time.
    - +1.0 for activating a contraption (like a button).
    - +0.1 for building a platform.
    - Small negative reward per step to encourage speed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a steampunk world by flipping gravity and building temporary platforms to reach the exit portal."
    )
    user_guide = (
        "Use ←→ to move and ↑ to jump. Press space to flip gravity and shift to build a platform."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1800 # 60 seconds * 30 FPS

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_BG_GEAR = (45, 40, 55)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 192, 255, 50)
    COLOR_PLATFORM = (139, 69, 19)
    COLOR_PLATFORM_EDGE = (101, 51, 14)
    COLOR_GOAL = (255, 215, 0)
    COLOR_GOAL_GLOW = (255, 215, 0, 60)
    COLOR_BUTTON = (200, 50, 50)
    COLOR_BUTTON_ACTIVE = (50, 200, 50)
    COLOR_DANGER = (255, 0, 0)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)

    # Physics
    GRAVITY_ACCEL = 0.8
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.85
    PLAYER_MAX_SPEED = 10
    PLAYER_JUMP_FORCE = 15.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        """Initializes all game state variables to a default value."""
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 30)
        self.gravity_vec = pygame.Vector2(0, self.GRAVITY_ACCEL)
        self.on_ground = False

        self.platforms = []
        self.built_platforms = []
        self.buttons = []
        self.goal = None

        self.level = 0
        self.platforms_to_build = 0
        self.platform_cooldown = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.background_gears = []
        
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.info = {}

    def _setup_level(self):
        """Sets up the environment for the current level."""
        self.built_platforms.clear()
        self.buttons.clear()
        self.particles.clear()
        self.platform_cooldown = 0
        
        # Level 0: Basic movement and goal
        if self.level == 0:
            self.player_pos = pygame.Vector2(100, 50)
            self.platforms = [
                pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40),
                pygame.Rect(300, 200, 200, 20)
            ]
            self.goal = pygame.Rect(400, 150, 40, 50)
            self.platforms_to_build = 0
        
        # Level 1: Platform building required
        elif self.level == 1:
            self.player_pos = pygame.Vector2(100, 50)
            self.platforms = [
                pygame.Rect(0, self.SCREEN_HEIGHT - 40, 250, 40),
                pygame.Rect(450, self.SCREEN_HEIGHT - 40, 200, 40)
            ]
            self.goal = pygame.Rect(500, self.SCREEN_HEIGHT - 120, 40, 80)
            self.platforms_to_build = 1

        # Level 2: Gravity flip required
        elif self.level == 2:
            self.player_pos = pygame.Vector2(100, 320)
            self.platforms = [
                pygame.Rect(0, 0, self.SCREEN_WIDTH, 40),
                pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40),
                pygame.Rect(200, 180, 240, 40)
            ]
            self.goal = pygame.Rect(300, 50, 40, 50)
            self.platforms_to_build = 0
        
        # Level 3: Button activation
        elif self.level == 3:
            self.player_pos = pygame.Vector2(50, 50)
            self.platforms = [
                pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
            ]
            # Button is a dict: {'rect': Rect, 'active': bool, 'linked_platform': Rect}
            self.buttons.append({
                'rect': pygame.Rect(550, 320, 30, 30),
                'active': False,
                'linked_platform': pygame.Rect(250, 200, 150, 20)
            })
            self.goal = pygame.Rect(315, 150, 40, 50)
            self.platforms_to_build = 0

        else: # Loop back to first level
            self.level = 0
            self._setup_level()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()
        
        if options and 'level' in options:
            self.level = options['level']
        
        self._setup_level()
        self.time_left = self.MAX_STEPS
        
        # Setup background elements
        if not self.background_gears:
            for _ in range(10):
                self.background_gears.append({
                    'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                    'radius': random.randint(20, 100),
                    'speed': random.uniform(-0.5, 0.5),
                    'angle': random.uniform(0, 360),
                    'teeth': random.randint(8, 20)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty for each step to encourage speed
        self.game_over = False

        # --- Handle one-shot actions ---
        space_just_pressed = space_pressed and not self.prev_space_held
        shift_just_pressed = shift_pressed and not self.prev_shift_held
        self.prev_space_held = space_pressed
        self.prev_shift_held = shift_pressed

        if space_just_pressed:
            self._flip_gravity()
            # sfx: whoosh sound
        
        if self.platform_cooldown > 0:
            self.platform_cooldown -= 1

        if shift_just_pressed and self.platforms_to_build > 0 and self.platform_cooldown == 0:
            if self._build_platform():
                reward += 0.1
                self.platforms_to_build -= 1
                self.platform_cooldown = 15 # 0.5s cooldown
                # sfx: metallic clank sound

        # --- Update Physics and Game State ---
        self._apply_forces(movement)
        self._update_player_position()
        
        # Check button presses
        for button in self.buttons:
            if not button['active'] and self.player_rect.colliderect(button['rect']):
                button['active'] = True
                self.platforms.append(button['linked_platform'])
                reward += 1.0
                self._create_particles(pygame.Vector2(button['rect'].center), 30, self.COLOR_BUTTON_ACTIVE)
                # sfx: click and activate sound

        self.time_left -= 1
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_rect.colliderect(self.goal):
            reward += 100
            terminated = True
            self.level += 1
            # sfx: level complete fanfare
        elif not self.screen.get_rect().colliderect(self.player_rect):
            reward -= 100
            terminated = True
            # sfx: falling scream
        elif self.time_left <= 0:
            reward -= 50
            terminated = True
            # sfx: timeout buzzer
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated always False
            self._get_info()
        )

    def _apply_forces(self, movement):
        # Apply gravity
        self.player_vel += self.gravity_vec

        # Apply movement force relative to gravity
        is_gravity_down = self.gravity_vec.y > 0
        
        # UP is always against gravity
        if movement == 1 and self.on_ground:
            jump_dir = -self.gravity_vec.normalize()
            self.player_vel += jump_dir * self.PLAYER_JUMP_FORCE
            # sfx: jump sound

        # DOWN is with gravity (no special action)
        # LEFT/RIGHT are perpendicular to gravity
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL

        # Apply friction and clamp velocity
        self.player_vel.x *= self.PLAYER_FRICTION
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))
        self.player_vel.y = max(-self.PLAYER_MAX_SPEED*2, min(self.PLAYER_MAX_SPEED*2, self.player_vel.y))

    def _update_player_position(self):
        self.on_ground = False
        all_platforms = self.platforms + self.built_platforms
        
        # Move horizontally
        self.player_pos.x += self.player_vel.x
        self.player_rect.centerx = int(self.player_pos.x)
        for plat in all_platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0: # Moving right
                    self.player_rect.right = plat.left
                elif self.player_vel.x < 0: # Moving left
                    self.player_rect.left = plat.right
                self.player_pos.x = self.player_rect.centerx
                self.player_vel.x = 0

        # Move vertically
        self.player_pos.y += self.player_vel.y
        self.player_rect.centery = int(self.player_pos.y)
        for plat in all_platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0: # Moving down
                    self.player_rect.bottom = plat.top
                    self.on_ground = True
                elif self.player_vel.y < 0: # Moving up
                    self.player_rect.top = plat.bottom
                    self.on_ground = True
                self.player_pos.y = self.player_rect.centery
                self.player_vel.y = 0

    def _flip_gravity(self):
        self.gravity_vec.y *= -1
        self.player_rect.height, self.player_rect.width = self.player_rect.width, self.player_rect.height # Visually rotate
        self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, is_shockwave=True)
    
    def _build_platform(self):
        # Build platform at player's feet, relative to gravity
        is_gravity_down = self.gravity_vec.y > 0
        if is_gravity_down:
            plat_pos_y = self.player_rect.bottom + 5
        else:
            plat_pos_y = self.player_rect.top - 25
        
        new_plat = pygame.Rect(self.player_rect.centerx - 40, plat_pos_y, 80, 20)

        # Prevent building inside other platforms
        for plat in self.platforms + self.built_platforms:
            if new_plat.colliderect(plat):
                return False

        self.built_platforms.append(new_plat)
        self._create_particles(pygame.Vector2(new_plat.center), 20, self.COLOR_PLATFORM)
        return True

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "level": self.level,
            "platforms_to_build": self.platforms_to_build
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for gear in self.background_gears:
            gear['angle'] += gear['speed']
            self._draw_gear(
                self.screen, self.COLOR_BG_GEAR, gear['pos'],
                gear['radius'], gear['teeth'], gear['angle']
            )

    def _render_game_elements(self):
        # Draw particles
        self._update_and_draw_particles()
        
        # Draw platforms
        for plat in self.platforms + self.built_platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, plat, 3)

        # Draw buttons
        for button in self.buttons:
            color = self.COLOR_BUTTON_ACTIVE if button['active'] else self.COLOR_BUTTON
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.COLOR_TEXT, button['rect'], 2)

        # Draw Goal
        if self.goal:
            anim_offset = math.sin(self.steps * 0.1) * 5
            glow_radius = self.goal.width / 2 + 15 + anim_offset
            self._draw_glow_circle(self.screen, self.goal.center, glow_radius, self.COLOR_GOAL_GLOW)
            pygame.draw.ellipse(self.screen, self.COLOR_GOAL, self.goal)
            
        # Draw Player
        glow_radius = max(self.player_rect.width, self.player_rect.height) * 1.2
        self._draw_glow_circle(self.screen, self.player_rect.center, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)
        
    def _render_ui(self):
        # Draw text with a shadow for readability
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Score and Level
        info_text = f"Level: {self.level + 1} | Score: {int(self.score)}"
        draw_text(info_text, self.font_ui, self.COLOR_TEXT, (10, 10))

        # Timer
        time_str = f"Time: {max(0, self.time_left // self.FPS):02d}"
        time_color = self.COLOR_DANGER if self.time_left < 10 * self.FPS else self.COLOR_TEXT
        draw_text(time_str, self.font_ui, time_color, (self.SCREEN_WIDTH - 150, 10))
        
        # Platforms available
        plat_str = f"Platforms: {self.platforms_to_build}"
        draw_text(plat_str, self.font_ui, self.COLOR_TEXT, (10, 35))

    def _create_particles(self, pos, count, color, is_shockwave=False):
        for _ in range(count):
            if is_shockwave:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 5)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
            
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': random.randint(15, 30),
                'color': color,
                'radius': random.randint(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)
            else:
                pos = (int(p['pos'].x), int(p['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    @staticmethod
    def _draw_glow_circle(surface, center, radius, color):
        if radius < 1: return
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        surface.blit(surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    @staticmethod
    def _draw_gear(surface, color, center, radius, num_teeth, angle_deg):
        angle_rad = math.radians(angle_deg)
        points = []
        tooth_depth = radius / 10
        for i in range(num_teeth * 2):
            r = radius if i % 2 == 0 else radius - tooth_depth
            angle = (i / (num_teeth * 2)) * 2 * math.pi + angle_rad
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((x, y))
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius - tooth_depth * 1.5), color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage and Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # This part requires a display. It will not run in a headless environment.
    # To run, comment out os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Clockwork City")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Arrows: Move")
        print("Space: Flip Gravity")
        print("Shift: Build Platform")
        print("R: Reset")
        print("Q: Quit")
        
        while not terminated:
            # --- Manual Control Mapping ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        terminated = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print(f"--- Level {info['level'] + 1} ---")

            # --- Step Environment ---
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if term:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print(f"Final Info: {info}")
                obs, info = env.reset()
                total_reward = 0
                print(f"\n--- Starting Level {info['level'] + 1} ---")
            
            # --- Render to Screen ---
            # The observation is (H, W, C), but pygame wants (W, H) surface
            # and surfarray.make_surface transposes it back.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"\nCould not run manual play example: {e}")
        print("This is expected in a headless environment. The GameEnv class is still valid.")
        env.close()