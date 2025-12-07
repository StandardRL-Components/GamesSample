import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:35:48.864247
# Source Brief: brief_01157.md
# Brief Index: 1157
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Spirit:
    def __init__(self, pos, target_pos, speed=5):
        self.pos = pygame.math.Vector2(pos)
        self.target_pos = pygame.math.Vector2(target_pos)
        self.vel = (self.target_pos - self.pos).normalize() * speed
        self.radius = 8
        self.color = (100, 255, 150) # Bright Green

    def update(self):
        self.pos += self.vel
        # Check if it has reached or passed the target
        if (self.pos - self.target_pos).length_squared() < (self.vel.length_squared() if self.vel.length() > 0 else 1):
             self.pos = self.target_pos
             return True # Reached destination
        return False

class Patrol:
    def __init__(self, pos, target_pos, speed=1.0):
        self.pos = pygame.math.Vector2(pos)
        self.target_pos = pygame.math.Vector2(target_pos)
        self.speed = speed
        self.radius = 10
        self.color = (255, 80, 80) # Bright Red
        self.update_velocity()

    def update_velocity(self):
        direction = self.target_pos - self.pos
        if direction.length() > 0:
            self.vel = direction.normalize() * self.speed
        else:
            self.vel = pygame.math.Vector2(0, 0)

    def update(self):
        self.pos += self.vel

class Particle:
    def __init__(self, pos, color, min_speed=1, max_speed=3, gravity=0.1, lifespan=30):
        self.pos = pygame.math.Vector2(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        self.vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = lifespan
        self.color = color
        self.gravity = gravity
        self.size = random.randint(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.lifespan -= 1
        return self.lifespan <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend the Sacred Grove by deploying spirits to intercept incoming enemies. "
        "Time your actions to the rhythm of the grove to protect its heart."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a portal. Press space on the beat "
        "to deploy a spirit and intercept enemies."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.CENTER = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("celtictime", 24)
        self.font_large = pygame.font.SysFont("celtictime", 48)

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GROVE = (255, 215, 0)
        self.COLOR_PORTAL_INACTIVE = (0, 100, 150)
        self.COLOR_PORTAL_SELECTED = (100, 200, 255)
        self.COLOR_PORTAL_ACTIVE = (200, 255, 255)
        self.COLOR_UI = (220, 220, 240)
        self.COLOR_PULSE = (40, 60, 90)

        # Game constants
        self.WIN_STEPS = 1000
        self.MAX_STEPS = 1200 # Absolute max
        self.GROVE_RADIUS = 35
        self.RHYTHM_PERIOD = 30 # A beat every second at 30fps
        self.SPIRIT_COOLDOWN = 60 # 2 seconds
        self.PATROL_SPAWN_INTERVAL = 150 # 5 seconds

        # Portal setup
        portal_dist = 150
        self.portal_positions = [
            self.CENTER + pygame.math.Vector2(0, -portal_dist), # Up
            self.CENTER + pygame.math.Vector2(0, portal_dist),  # Down
            self.CENTER + pygame.math.Vector2(-portal_dist, 0), # Left
            self.CENTER + pygame.math.Vector2(portal_dist, 0)   # Right
        ]
        self.portal_radius = 20

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pending_reward = 0.0

        # Game entities
        self.spirits = []
        self.patrols = []
        self.particles = []
        self.active_portals = [False] * 4 # Tracks if a portal has a spirit guarding it

        # Player state
        self.selected_portal_idx = 0
        self.last_space_state = False
        self.last_movement_action = 0

        # Rhythm and cooldowns
        self.beat_counter = 0
        self.spirit_recharge_timer = 0
        self.next_patrol_spawn = self.PATROL_SPAWN_INTERVAL

        # Difficulty progression
        self.base_patrol_speed = 1.0
        self.max_patrols = 1

        self._spawn_patrol()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, _ = action
        space_pressed = space_action == 1 and not self.last_space_state
        self.last_space_state = space_action == 1

        # 1. Update Game Logic
        self.steps += 1
        self.beat_counter = (self.beat_counter + 1) % self.RHYTHM_PERIOD
        self.pending_reward = 0.0

        self._update_difficulty()
        self._update_player_input(movement, space_pressed)
        self._update_timers_and_spawns()
        self._update_spirits()
        self._update_patrols()
        self._handle_collisions()
        self._update_particles()

        # 2. Calculate Reward & Termination
        reward = 0.1 + self.pending_reward # Base survival reward + event rewards
        terminated = self.game_over
        truncated = False

        if any(p.pos.distance_to(self.CENTER) < self.GROVE_RADIUS for p in self.patrols):
            terminated = True
            reward = -100.0
        elif self.steps >= self.WIN_STEPS:
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
             terminated = True
             truncated = True


        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_input(self, movement, space_pressed):
        # Update selected portal only on a new movement command
        if movement != 0 and movement != self.last_movement_action:
            # 1=up, 2=down, 3=left, 4=right
            if movement == 1: self.selected_portal_idx = 0
            elif movement == 2: self.selected_portal_idx = 1
            elif movement == 3: self.selected_portal_idx = 2
            elif movement == 4: self.selected_portal_idx = 3
        self.last_movement_action = movement

        # Deploy spirit on space press if cooldown is over and beat is right
        is_on_beat = self.beat_counter == 0
        if space_pressed and self.spirit_recharge_timer == 0 and is_on_beat:
            # sfx: SpiritDeploy.wav
            target_pos = self.portal_positions[self.selected_portal_idx]
            self.spirits.append(Spirit(self.CENTER, target_pos))
            self.spirit_recharge_timer = self.SPIRIT_COOLDOWN
            for _ in range(20):
                self.particles.append(Particle(self.CENTER, self.COLOR_PORTAL_ACTIVE, max_speed=4))


    def _update_difficulty(self):
        self.base_patrol_speed = 1.0 + 0.05 * (self.steps // 200)
        self.max_patrols = min(5, 1 + (self.steps // 300))

    def _update_timers_and_spawns(self):
        if self.spirit_recharge_timer > 0:
            self.spirit_recharge_timer -= 1
        if self.steps >= self.next_patrol_spawn and len(self.patrols) < self.max_patrols:
            self._spawn_patrol()
            self.next_patrol_spawn += self.PATROL_SPAWN_INTERVAL

    def _update_spirits(self):
        for spirit in self.spirits[:]:
            if spirit.update(): # Reached destination
                self.spirits.remove(spirit)
                # For simplicity, spirits disappear on arrival. A more complex game
                # might have them "guard" the portal for a duration.

    def _update_patrols(self):
        for patrol in self.patrols:
            patrol.speed = self.base_patrol_speed
            patrol.update_velocity()
            patrol.update()

    def _handle_collisions(self):
        for spirit in self.spirits[:]:
            for patrol in self.patrols[:]:
                if spirit.pos.distance_to(patrol.pos) < spirit.radius + patrol.radius:
                    # sfx: Intercept.wav
                    self.pending_reward += 1.0
                    self.score += 10
                    for _ in range(50):
                        self.particles.append(Particle(patrol.pos, patrol.color))
                    self.spirits.remove(spirit)
                    self.patrols.remove(patrol)
                    break # Spirit can only hit one patrol

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _spawn_patrol(self):
        # sfx: RomanMarch.wav
        edge = random.randint(0, 3)
        if edge == 0: # Top
            pos = (random.randint(0, self.WIDTH), -20)
        elif edge == 1: # Bottom
            pos = (random.randint(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 2: # Left
            pos = (-20, random.randint(0, self.HEIGHT))
        else: # Right
            pos = (self.WIDTH + 20, random.randint(0, self.HEIGHT))
        self.patrols.append(Patrol(pos, self.CENTER, self.base_patrol_speed))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Rhythm Pulse
        pulse_alpha = 1 - (self.beat_counter / self.RHYTHM_PERIOD)
        pulse_radius = int(self.GROVE_RADIUS + 200 * (1 - pulse_alpha))
        pulse_color = (*self.COLOR_PULSE, int(64 * pulse_alpha**2))
        self._draw_aa_circle(self.screen, pulse_color, self.CENTER, pulse_radius, 2)

        # Sacred Grove
        self._draw_celtic_knot(self.CENTER, self.GROVE_RADIUS, self.COLOR_GROVE, 4)

        # Portals
        for i, pos in enumerate(self.portal_positions):
            color = self.COLOR_PORTAL_INACTIVE
            if i == self.selected_portal_idx:
                color = self.COLOR_PORTAL_SELECTED
                # Draw selector arrow
                angle = (pos - self.CENTER).angle_to(pygame.math.Vector2(0, -1))
                p1 = self.CENTER + (pos - self.CENTER).normalize() * (self.GROVE_RADIUS + 15)
                p2 = p1 + pygame.math.Vector2(0, 10).rotate(-angle)
                p3 = p1 + pygame.math.Vector2(0, -10).rotate(-angle)
                pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3], color)
                pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3], color)

            self._draw_aa_circle(self.screen, color, pos, self.portal_radius, 2)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, p.pos, p.size)

        # Spirits
        for spirit in self.spirits:
            self._draw_glowing_circle(spirit.pos, spirit.color, spirit.radius)

        # Patrols
        for patrol in self.patrols:
            self._draw_glowing_circle(patrol.pos, patrol.color, patrol.radius)
            # Draw a small triangle "helmet" on top
            p1 = patrol.pos + pygame.math.Vector2(0, -patrol.radius)
            p2 = patrol.pos + pygame.math.Vector2(-patrol.radius*0.6, -patrol.radius*1.5)
            p3 = patrol.pos + pygame.math.Vector2(patrol.radius*0.6, -patrol.radius*1.5)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], patrol.color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], patrol.color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Steps / Time
        time_text = self.font_small.render(f"TIME: {self.steps} / {self.WIN_STEPS}", True, self.COLOR_UI)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Spirit Cooldown Bar
        bar_width = 200
        bar_height = 15
        bar_x = self.CENTER.x - bar_width / 2
        bar_y = self.HEIGHT - bar_height - 10
        
        # Cooldown progress
        fill_ratio = 1.0 - (self.spirit_recharge_timer / self.SPIRIT_COOLDOWN) if self.SPIRIT_COOLDOWN > 0 else 1.0
        fill_width = int(bar_width * fill_ratio)

        pygame.draw.rect(self.screen, self.COLOR_PULSE, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PORTAL_SELECTED, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # Add "READY" text when full
        if fill_ratio >= 1.0:
            ready_text = self.font_small.render("READY", True, self.COLOR_UI)
            self.screen.blit(ready_text, (self.CENTER.x - ready_text.get_width() / 2, bar_y - ready_text.get_height() - 2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "patrols": len(self.patrols),
            "spirits": len(self.spirits),
        }

    def _draw_aa_circle(self, surface, color, center, radius, width=0):
        if radius > 0:
            center_int = (int(center.x), int(center.y))
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
            if width > 1:
                pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius-width+1), color)

    def _draw_glowing_circle(self, pos, color, radius):
        pos_int = (int(pos.x), int(pos.y))
        
        # Create a temporary surface for the glow
        glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        
        # Draw concentric circles with decreasing alpha for the glow
        for i in range(int(radius), 0, -2):
            alpha = int(80 * (i / radius)**2)
            pygame.gfxdraw.filled_circle(glow_surf, int(radius*2), int(radius*2), i + int(radius), (*color, alpha))

        # Blit the glow surface centered on the position
        self.screen.blit(glow_surf, (pos_int[0] - radius*2, pos_int[1] - radius*2))
        
        # Draw the solid core on top
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius), color)

    def _draw_celtic_knot(self, center, radius, color, width):
        # Simplified knot effect using overlapping circles
        num_circles = 8
        for i in range(num_circles):
            angle = i * (2 * math.pi / num_circles)
            offset_angle = math.sin(i * 4) * 0.3 # Weave in and out
            offset_radius = radius * 0.7 + radius * 0.2 * offset_angle
            pos = center + pygame.math.Vector2(offset_radius, 0).rotate_rad(angle)
            self._draw_aa_circle(self.screen, color, pos, int(radius * 0.4), width)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    # The following line is needed to render in a window for human play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Grove Guardian")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Manual control mapping
        movement = 0 # 0: none
        space_held = 0 # 0: released
        shift_held = 0 # 0: released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(30) # Run at 30 FPS

    env.close()