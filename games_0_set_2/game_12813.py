import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:00:26.574811
# Source Brief: brief_02813.md
# Brief Index: 2813
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a high-tech ship through a dangerous gluon field. "
        "Collect energy, manage your systems, and race through checkpoints to achieve the highest score."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press Shift to cycle between Normal, Shield, and Portal modes. "
        "Press Space to activate the current mode's ability (Shield or Portal)."
    )
    auto_advance = True
    
    # Game states
    STATE_NORMAL = 0
    STATE_SHIELD_READY = 1
    STATE_PORTAL_READY = 2

    # Visual Constants
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (10, 0, 20)
    COLOR_GRID = (30, 10, 50)
    COLOR_PLAYER_NORMAL = (0, 150, 255)
    COLOR_PLAYER_SHIELD = (0, 255, 150)
    COLOR_PLAYER_PORTAL = (200, 0, 255)
    COLOR_GLUON_DANGER = (255, 50, 50)
    COLOR_ENERGY_ORB = (255, 255, 0)
    COLOR_CHECKPOINT = (200, 200, 200)
    COLOR_CHECKPOINT_ACTIVE = (100, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)

    # Gameplay Constants
    MAX_STEPS = 5000
    PLAYER_ACCELERATION = 0.8
    PLAYER_FRICTION = 0.92
    PLAYER_MAX_SPEED = 8.0
    PLAYER_SIZE = 12
    INITIAL_ENERGY = 50.0
    MAX_ENERGY = 100.0
    ENERGY_ORB_VALUE = 25.0
    SHIELD_ENERGY_DRAIN = 0.4
    PORTAL_ENERGY_COST = 30.0
    PORTAL_DISTANCE = 150
    GLUON_DAMAGE = 1.0
    CHECKPOINT_RADIUS = 30
    ENERGY_ORB_RADIUS = 8
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_energy = 0.0
        self.player_state = self.STATE_NORMAL
        self.is_shield_active = False
        self.last_shift_held = False
        self.last_space_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.checkpoints = []
        self.current_checkpoint_idx = 0
        self.energy_orbs = []
        self.gluon_zones = []
        self.gluon_fluctuation_freq = 0.5
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.WIDTH * 0.1, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_energy = self.INITIAL_ENERGY
        self.player_state = self.STATE_NORMAL
        self.is_shield_active = False
        self.last_shift_held = False
        self.last_space_held = False
        
        self.checkpoints = [
            pygame.math.Vector2(self.WIDTH * 0.3, self.HEIGHT * 0.25),
            pygame.math.Vector2(self.WIDTH * 0.5, self.HEIGHT * 0.75),
            pygame.math.Vector2(self.WIDTH * 0.7, self.HEIGHT * 0.25),
            pygame.math.Vector2(self.WIDTH * 0.9, self.HEIGHT * 0.5), # Final
        ]
        self.current_checkpoint_idx = 0
        self._spawn_energy_orbs()

        self.gluon_zones = []
        self.gluon_fluctuation_freq = 0.5
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        self._handle_input(action)
        self._update_player_movement()
        self._update_gluon_field()
        
        collision_reward = self._handle_collisions_and_interactions()
        reward += collision_reward

        self.player_energy = max(0, self.player_energy)
        self.steps += 1
        
        # Termination conditions
        if self.player_energy <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
        
        if self.current_checkpoint_idx >= len(self.checkpoints):
            terminated = True
            reward += 100
            self.score += 1000
            self.game_over = True # Win condition

        if self.steps >= self.MAX_STEPS:
            terminated = True # Use terminated for time limit, truncated is for external conditions
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y = -self.PLAYER_ACCELERATION
        elif movement == 2: accel.y = self.PLAYER_ACCELERATION
        elif movement == 3: accel.x = -self.PLAYER_ACCELERATION
        elif movement == 4: accel.x = self.PLAYER_ACCELERATION
        self.player_vel += accel

        # State switching (on press)
        if shift_held and not self.last_shift_held:
            self.player_state = (self.player_state + 1) % 3
            # sfx: state_switch.wav
            self._create_particles(self.player_pos, 15, self._get_player_color(), 2, 20)

        # Abilities
        self.is_shield_active = False
        if space_held:
            if self.player_state == self.STATE_SHIELD_READY and self.player_energy > 0:
                self.is_shield_active = True
                self.player_energy -= self.SHIELD_ENERGY_DRAIN
                # sfx: shield_loop.wav
            elif self.player_state == self.STATE_PORTAL_READY and not self.last_space_held:
                if self.player_energy >= self.PORTAL_ENERGY_COST:
                    self.player_energy -= self.PORTAL_ENERGY_COST
                    start_pos = self.player_pos.copy()
                    self.player_pos.x += self.PORTAL_DISTANCE
                    self._create_particles(start_pos, 30, self.COLOR_PLAYER_PORTAL, 4, 30)
                    self._create_particles(self.player_pos, 30, self.COLOR_PLAYER_PORTAL, 4, 30)
                    # sfx: portal_activate.wav

        self.last_shift_held = shift_held
        self.last_space_held = space_held

    def _update_player_movement(self):
        # Limit speed
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Update position
        self.player_pos += self.player_vel
        
        # Clamp to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Thruster particles
        if self.player_vel.length() > 1:
            angle = self.player_vel.angle_to(pygame.math.Vector2(1, 0)) + 180
            p_pos = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0).rotate(-angle)
            self._create_particles(p_pos, 1, self.COLOR_PLAYER_NORMAL, 1, 15, angle_spread=20)


    def _update_gluon_field(self):
        if self.steps % 500 == 0 and self.steps > 0:
            self.gluon_fluctuation_freq = min(2.0, self.gluon_fluctuation_freq + 0.05)

        self.gluon_zones = []
        num_zones = 3 + int(self.steps / 1000)
        for i in range(num_zones):
            t = self.steps * self.gluon_fluctuation_freq * 0.1
            x = self.WIDTH * (0.5 + 0.4 * math.sin(t + i * 2.1))
            y = self.HEIGHT * (0.5 + 0.4 * math.cos(t * 0.7 + i * 1.5))
            radius = 30 + 20 * (1 + math.sin(t * 1.3 + i))
            self.gluon_zones.append({'pos': pygame.math.Vector2(x, y), 'radius': radius})

    def _handle_collisions_and_interactions(self):
        reward = 0
        
        # Energy Orbs
        for orb in self.energy_orbs[:]:
            if self.player_pos.distance_to(orb) < self.PLAYER_SIZE + self.ENERGY_ORB_RADIUS:
                self.energy_orbs.remove(orb)
                self.player_energy = min(self.MAX_ENERGY, self.player_energy + self.ENERGY_ORB_VALUE)
                reward += 0.1
                self.score += 10
                self._create_particles(orb, 20, self.COLOR_ENERGY_ORB, 2, 25)
                # sfx: collect_orb.wav

        # Gluon Field
        in_danger_zone = False
        for zone in self.gluon_zones:
            if self.player_pos.distance_to(zone['pos']) < zone['radius'] + self.PLAYER_SIZE:
                in_danger_zone = True
                break
        
        if in_danger_zone and not self.is_shield_active:
            self.player_energy -= self.GLUON_DAMAGE
            reward -= 0.01
            # sfx: damage.wav
            if random.random() < 0.2:
                self._create_particles(self.player_pos, 3, self.COLOR_GLUON_DANGER, 1, 10)

        # Checkpoints
        if self.current_checkpoint_idx < len(self.checkpoints):
            target = self.checkpoints[self.current_checkpoint_idx]
            if self.player_pos.distance_to(target) < self.CHECKPOINT_RADIUS:
                self.current_checkpoint_idx += 1
                reward += 1
                self.score += 100
                self._spawn_energy_orbs()
                self._create_particles(target, 50, self.COLOR_CHECKPOINT_ACTIVE, 3, 40)
                # sfx: checkpoint.wav
        
        return reward
    
    def _spawn_energy_orbs(self):
        self.energy_orbs.clear()
        for _ in range(5):
            x = random.randint(int(self.WIDTH*0.1), int(self.WIDTH*0.9))
            y = random.randint(int(self.HEIGHT*0.1), int(self.HEIGHT*0.9))
            self.energy_orbs.append(pygame.math.Vector2(x, y))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background_grid()
        self._render_gluon_field()
        self._render_checkpoints()
        self._render_energy_orbs()
        self._render_particles()
        self._render_player()

    def _render_background_grid(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_gluon_field(self):
        for zone in self.gluon_zones:
            pos = (int(zone['pos'].x), int(zone['pos'].y))
            radius = int(zone['radius'])
            if radius <= 0: continue
            
            # Glow effect
            alpha = 50 + 25 * math.sin(self.steps * 0.1)
            glow_color = (*self.COLOR_GLUON_DANGER, alpha)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_GLUON_DANGER)

    def _render_checkpoints(self):
        for i, pos in enumerate(self.checkpoints):
            color = self.COLOR_CHECKPOINT_ACTIVE if i == self.current_checkpoint_idx else self.COLOR_CHECKPOINT
            radius = self.CHECKPOINT_RADIUS
            
            # Pulsating glow for active checkpoint
            if i == self.current_checkpoint_idx:
                glow_radius = radius + 10 * (1 + math.sin(self.steps * 0.1))
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(glow_radius), (*color, 20))
            
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), radius - 2, color)


    def _render_energy_orbs(self):
        for orb in self.energy_orbs:
            pos = (int(orb.x), int(orb.y))
            radius = self.ENERGY_ORB_RADIUS
            # Glow effect
            glow_radius = int(radius * (1.5 + 0.5 * math.sin(self.steps * 0.2)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_ENERGY_ORB, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENERGY_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,255,255))

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        color = self._get_player_color()
        angle_rad = math.atan2(self.player_vel.y, self.player_vel.x) if self.player_vel.length() > 0 else 0

        # Triangle points
        p1 = (pos[0] + self.PLAYER_SIZE * math.cos(angle_rad), pos[1] + self.PLAYER_SIZE * math.sin(angle_rad))
        p2 = (pos[0] + self.PLAYER_SIZE * math.cos(angle_rad + 2.356), pos[1] + self.PLAYER_SIZE * math.sin(angle_rad + 2.356))
        p3 = (pos[0] + self.PLAYER_SIZE * math.cos(angle_rad - 2.356), pos[1] + self.PLAYER_SIZE * math.sin(angle_rad - 2.356))
        points = [p1, p2, p3]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        
        # Glow
        pygame.gfxdraw.filled_trigon(self.screen, *int_points[0], *int_points[1], *int_points[2], (*color, 60))
        # Body
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        
        # Shield effect
        if self.is_shield_active:
            radius = int(self.PLAYER_SIZE * (2.2 + 0.2 * math.sin(self.steps * 0.3)))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER_SHIELD)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*self.COLOR_PLAYER_SHIELD, 30))

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = (*p['color'], alpha)
                radius = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
                if radius > 0:
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(p['pos'].x-radius), int(p['pos'].y-radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Energy Bar
        bar_width = 200
        bar_height = 20
        energy_ratio = self.player_energy / self.MAX_ENERGY
        fill_width = int(bar_width * energy_ratio)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self._get_player_color(), (10, 10, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)
        
        # Speed Text
        speed_text = f"SPD: {self.player_vel.length():.1f}"
        text_surf = self.font_small.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # State Text
        state_names = {0: "NORMAL", 1: "SHIELD", 2: "PORTAL"}
        state_text = f"MODE: {state_names[self.player_state]}"
        text_surf = self.font_small.render(state_text, True, self._get_player_color())
        self.screen.blit(text_surf, (10, 35))

        # Score Text
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 35))

    def _get_player_color(self):
        if self.player_state == self.STATE_NORMAL: return self.COLOR_PLAYER_NORMAL
        if self.player_state == self.STATE_SHIELD_READY: return self.COLOR_PLAYER_SHIELD
        if self.player_state == self.STATE_PORTAL_READY: return self.COLOR_PLAYER_PORTAL
        return (255, 255, 255)
    
    def _create_particles(self, pos, count, color, speed, lifespan, angle_spread=360):
        for _ in range(count):
            angle = random.uniform(-angle_spread/2, angle_spread/2)
            vel_angle = math.radians(random.uniform(0, 360) if angle_spread == 360 else angle)
            p_vel = pygame.math.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * random.uniform(0.5, 1) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': p_vel,
                'lifespan': random.randint(lifespan // 2, lifespan),
                'max_lifespan': lifespan,
                'color': color,
                'size': random.uniform(1, 4)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.player_energy,
            "player_state": self.player_state,
            "checkpoint": self.current_checkpoint_idx
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # The main loop is for human play and visualization, not part of the gym env
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gluon Racer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # To control with keyboard
    keys_pressed = {
        pygame.K_UP: False, pygame.K_DOWN: False,
        pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False
    }

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------\n")
    
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_pressed:
                    keys_pressed[event.key] = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
            if event.type == pygame.KEYUP:
                if event.key in keys_pressed:
                    keys_pressed[event.key] = False

        if not terminated:
            # Map keyboard to action space
            movement = 0 # none
            if keys_pressed[pygame.K_UP]: movement = 1
            elif keys_pressed[pygame.K_DOWN]: movement = 2
            elif keys_pressed[pygame.K_LEFT]: movement = 3
            elif keys_pressed[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys_pressed[pygame.K_SPACE] else 0
            shift = 1 if keys_pressed[pygame.K_LSHIFT] else 0
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

        if terminated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for reset
            
    env.close()