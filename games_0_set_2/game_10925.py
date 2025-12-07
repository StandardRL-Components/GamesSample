import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:12:28.353193
# Source Brief: brief_00925.md
# Brief Index: 925
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to collect green orbs while avoiding red orbs by switching
    gravity and navigating moving platforms.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up(unused), 2=down(unused), 3=left, 4=right)
    - action[1]: Gravity Switch (0=released, 1=pressed)
    - action[2]: Shift (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball by switching gravity. Collect green orbs for points "
        "and avoid red orbs and falling off-screen."
    )
    user_guide = "Use the ←→ arrow keys to move. Press the space bar to switch the direction of gravity."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_GREEN_ORB = (0, 255, 150)
    COLOR_RED_ORB = (255, 50, 50)
    COLOR_PLATFORM = (100, 110, 120)
    COLOR_TRANSFORM_PLATFORM = (100, 200, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    # Physics
    PLAYER_RADIUS = 12
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.08
    GRAVITY_ACCEL = 0.35
    BOUNCE_DAMPENING = 0.85
    ORB_RADIUS = 7

    # Game Rules
    MAX_STEPS = 1500 # 50 seconds at 30fps
    WIN_SCORE = 20
    LOSE_SCORE = -10
    NUM_GREEN_ORBS = 5
    NUM_RED_ORBS = 3
    NUM_OSC_PLATFORMS = 3
    NUM_TRANSFORM_PLATFORMS = 2
    TRANSFORM_PLATFORM_CYCLE = 5 * FPS # 5 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        self.render_mode = render_mode
        self._bg_surface = self._create_background()

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.gravity_direction = None
        self.steps = None
        self.score = None
        self.green_orbs_collected = None
        self.game_over = None
        self.last_space_held = None
        self.platforms = None
        self.orbs = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.gravity_direction = 1  # 1 for down, -1 for up
        
        self.steps = 0
        self.score = 0
        self.green_orbs_collected = 0
        self.game_over = False
        self.last_space_held = False
        
        self._create_platforms()
        self._create_orbs()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Small reward for surviving

        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
            
        # Gravity switch on press (not hold)
        if space_held and not self.last_space_held:
            self.gravity_direction *= -1
            # SFX: play_gravity_switch_sound()
        self.last_space_held = space_held

        # 2. Update Game Logic
        self._update_player_physics()
        self._update_platforms()
        
        # 3. Handle Collisions and Collect Orbs
        reward += self._handle_orb_collisions()
        self._handle_platform_collisions()
        self._handle_wall_collisions()

        # 4. Check for Termination
        terminated, term_reward = self._check_termination()
        self.game_over = terminated
        reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_physics(self):
        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY_ACCEL * self.gravity_direction
        
        # Update position
        self.player_pos += self.player_vel

    def _update_platforms(self):
        for p in self.platforms:
            p.update()

    def _handle_orb_collisions(self):
        reward = 0
        for orb in self.orbs[:]:
            dist = self.player_pos.distance_to(orb.pos)
            if dist < self.PLAYER_RADIUS + self.ORB_RADIUS:
                if orb.is_green:
                    self.score += 1
                    self.green_orbs_collected += 1
                    reward += 10.0
                    # SFX: play_collect_green_sound()
                else:
                    self.score -= 3
                    reward -= 5.0
                    # SFX: play_collect_red_sound()
                
                self.orbs.remove(orb)
                self._spawn_orb(orb.is_green)
        return reward

    def _handle_platform_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_RADIUS,
            self.player_pos.y - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )

        for p in self.platforms:
            if p.is_solid() and p.rect.colliderect(player_rect):
                # Vertical collision
                if (self.player_vel.y > 0 and 
                    player_rect.bottom > p.rect.top and 
                    player_rect.centery < p.rect.top):
                    self.player_pos.y = p.rect.top - self.PLAYER_RADIUS
                    self.player_vel.y *= -self.BOUNCE_DAMPENING
                    # SFX: play_bounce_sound()
                elif (self.player_vel.y < 0 and 
                      player_rect.top < p.rect.bottom and
                      player_rect.centery > p.rect.bottom):
                    self.player_pos.y = p.rect.bottom + self.PLAYER_RADIUS
                    self.player_vel.y *= -self.BOUNCE_DAMPENING
                    # SFX: play_bounce_sound()
                
                # Horizontal collision (less bouncy)
                if (self.player_vel.x > 0 and 
                    player_rect.right > p.rect.left and 
                    player_rect.centerx < p.rect.left):
                    self.player_pos.x = p.rect.left - self.PLAYER_RADIUS
                    self.player_vel.x *= -0.5
                elif (self.player_vel.x < 0 and 
                      player_rect.left < p.rect.right and
                      player_rect.centerx > p.rect.right):
                    self.player_pos.x = p.rect.right + self.PLAYER_RADIUS
                    self.player_vel.x *= -0.5

    def _handle_wall_collisions(self):
        # Bounce off side walls
        if self.player_pos.x - self.PLAYER_RADIUS < 0:
            self.player_pos.x = self.PLAYER_RADIUS
            self.player_vel.x *= -0.8
        elif self.player_pos.x + self.PLAYER_RADIUS > self.WIDTH:
            self.player_pos.x = self.WIDTH - self.PLAYER_RADIUS
            self.player_vel.x *= -0.8

    def _check_termination(self):
        # Win condition
        if self.score >= self.WIN_SCORE:
            return True, 100.0

        # Loss conditions
        if self.score <= self.LOSE_SCORE:
            return True, -100.0
        if not ( -self.PLAYER_RADIUS < self.player_pos.y < self.HEIGHT + self.PLAYER_RADIUS):
            return True, -100.0
            
        return False, 0.0

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "green_orbs": self.green_orbs_collected
        }

    def _render_frame(self):
        self.screen.blit(self._bg_surface, (0, 0))
        self._render_game_elements()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
    
    def _create_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_game_elements(self):
        for p in self.platforms:
            p.draw(self.screen)
        for orb in self.orbs:
            orb.draw(self.screen)
        self._render_player()

    def _render_player(self):
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - glow_radius)))
        
        # Player circle
        pygame.draw.circle(
            self.screen, self.COLOR_PLAYER, 
            (int(self.player_pos.x), int(self.player_pos.y)), 
            self.PLAYER_RADIUS
        )

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        orbs_text = self.font_ui.render(f"ORBS: {self.green_orbs_collected}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(orbs_text, (self.WIDTH - orbs_text.get_width() - 10, 10))
        
        # Gravity indicator
        indicator_y = 25 if self.gravity_direction == 1 else self.HEIGHT - 25
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [
            (self.WIDTH / 2 - 10, indicator_y - 5 * self.gravity_direction),
            (self.WIDTH / 2 + 10, indicator_y - 5 * self.gravity_direction),
            (self.WIDTH / 2, indicator_y + 5 * self.gravity_direction)
        ])

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        message = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
        text_surf = self.font_game_over.render(message, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _create_platforms(self):
        self.platforms = []
        platform_y_spacing = self.HEIGHT / (self.NUM_OSC_PLATFORMS + self.NUM_TRANSFORM_PLATFORMS + 1)
        
        for i in range(self.NUM_OSC_PLATFORMS):
            y = platform_y_spacing * (i + 1)
            self.platforms.append(OscillatingPlatform(
                x=random.uniform(0, self.WIDTH - 150),
                y=y,
                color=self.COLOR_PLATFORM
            ))
        
        for i in range(self.NUM_TRANSFORM_PLATFORMS):
            y = platform_y_spacing * (self.NUM_OSC_PLATFORMS + i + 1)
            self.platforms.append(TransformingPlatform(
                x=random.uniform(0, self.WIDTH - 150),
                y=y,
                cycle_time=self.TRANSFORM_PLATFORM_CYCLE,
                color=self.COLOR_TRANSFORM_PLATFORM
            ))

    def _create_orbs(self):
        self.orbs = []
        for _ in range(self.NUM_GREEN_ORBS):
            self._spawn_orb(is_green=True)
        for _ in range(self.NUM_RED_ORBS):
            self._spawn_orb(is_green=False)
            
    def _spawn_orb(self, is_green):
        while True:
            pos = pygame.Vector2(
                random.uniform(self.ORB_RADIUS, self.WIDTH - self.ORB_RADIUS),
                random.uniform(self.ORB_RADIUS, self.HEIGHT - self.ORB_RADIUS)
            )
            
            # Ensure orbs don't spawn inside platforms
            too_close = False
            for p in self.platforms:
                if p.rect.collidepoint(pos):
                    too_close = True
                    break
            if not too_close:
                self.orbs.append(Orb(pos, is_green, self.ORB_RADIUS, self.COLOR_GREEN_ORB, self.COLOR_RED_ORB))
                break

    def close(self):
        pygame.quit()

# --- Helper Classes ---

class Orb:
    def __init__(self, pos, is_green, radius, green_color, red_color):
        self.pos = pos
        self.is_green = is_green
        self.radius = radius
        self.color = green_color if is_green else red_color
    
    def draw(self, surface):
        # Glow
        glow_radius = int(self.radius * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, 80), (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)))
        # Orb
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

class Platform:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
    
    def update(self):
        pass
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
    
    def is_solid(self):
        return True

class OscillatingPlatform(Platform):
    def __init__(self, x, y, color):
        super().__init__(x, y, 120, 15, color)
        self.speed = random.uniform(1.0, 2.0) * random.choice([-1, 1])

    def update(self):
        self.rect.x += self.speed
        if self.rect.left < 0 or self.rect.right > GameEnv.WIDTH:
            self.speed *= -1
            self.rect.x = max(0, min(self.rect.x, GameEnv.WIDTH - self.rect.width))

class TransformingPlatform(Platform):
    def __init__(self, x, y, cycle_time, color):
        super().__init__(x, y, 150, 15, color)
        self.cycle_time = cycle_time
        self.timer = random.randint(0, cycle_time)
        self.solid = True
        self.blink_time = GameEnv.FPS # 1 second

    def update(self):
        self.timer -= 1
        if self.timer <= 0:
            self.solid = not self.solid
            self.timer = self.cycle_time
            # SFX: play_transform_sound()

    def is_solid(self):
        return self.solid

    def draw(self, surface):
        is_blinking = self.timer < self.blink_time and (self.timer // 6) % 2 == 0
        
        if self.solid:
            if is_blinking:
                # Blink to indicate impending change
                temp_color = (min(255, self.color[0]+50), min(255, self.color[1]+50), min(255, self.color[2]+50))
                pygame.draw.rect(surface, temp_color, self.rect, border_radius=3)
            else:
                pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
        else:
            # Draw as a semi-transparent outline
            alpha_color = (*self.color, 100)
            shape_surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, alpha_color, shape_surf.get_rect(), border_radius=3)
            surface.blit(shape_surf, self.rect.topleft)


# Example of how to run the environment with random actions
if __name__ == '__main__':
    # The following is for human play and visualization, and is not part of the
    # required headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv()
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Gravity Bounce")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        movement_action = 0 # none
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()

        action = [movement_action, space_action, 0] # [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()