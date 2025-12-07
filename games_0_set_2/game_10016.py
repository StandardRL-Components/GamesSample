import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
from dataclasses import dataclass
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@dataclass
class Satellite:
    id: int
    pos: pygame.Vector2
    vel: pygame.Vector2
    color: tuple[int, int, int]
    target_speed: float
    alive: bool = True
    radius: int = 7


@dataclass
class DebrisField:
    base_radius: float
    width: float
    amplitude: float
    frequency: float
    phase: float
    current_radius: float = 0.0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    game_description = (
        "Protect a constellation of satellites by adjusting their orbits to avoid hazardous debris fields."
    )
    user_guide = (
        "Use ↑/↓ to select a satellite. Hold Space to increase its speed (move to a higher orbit) "
        "and Shift to decrease its speed (move to a lower orbit)."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CENTER = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Game Parameters
        self.NUM_SATELLITES = 7
        self.MAX_STEPS = 5400  # 90 seconds * 60 FPS
        self.TARGET_ORBIT_RADIUS = 150
        self.ORBIT_RADIUS_TOLERANCE = 15
        self.MIN_SAT_SPEED = 2.0
        self.MAX_SAT_SPEED = 6.0
        self.SPEED_ADJUST_RATE = 0.05

        # Physics Constants (tuned for gameplay)
        self.G_CONST = 1000

        # Colors
        self.COLOR_BG = (13, 17, 23) # Dark blue/black
        self.COLOR_STAR = (255, 223, 186)
        self.COLOR_STAR_GLOW = (248, 196, 113)
        self.COLOR_TARGET_ORBIT = (45, 133, 84, 100) # Green, semi-transparent
        self.COLOR_DEBRIS = (248, 81, 73, 150) # Red, semi-transparent
        self.COLOR_TEXT = (201, 209, 217)
        self.COLOR_SELECTION = (255, 255, 255)
        self.SAT_COLORS = [
            (56, 182, 255), (16, 185, 129), (249, 115, 22), (219, 39, 119),
            (139, 92, 246), (239, 68, 68), (245, 208, 254)
        ]
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 12, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.satellites = []
        self.debris_fields = []
        self.selected_satellite_idx = 0
        self.action_cooldowns = {'selection': 0}
        
        # Particle effects
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.selected_satellite_idx = 0
        self.particles.clear()
        
        # Initialize Satellites
        self.satellites.clear()
        for i in range(self.NUM_SATELLITES):
            angle = (2 * math.pi / self.NUM_SATELLITES) * i
            pos = self.CENTER + pygame.Vector2(self.TARGET_ORBIT_RADIUS, 0).rotate_rad(angle)
            
            # Velocity for a stable circular orbit: v = sqrt(GM/r)
            initial_speed = math.sqrt(self.G_CONST / self.TARGET_ORBIT_RADIUS)
            vel = (pos - self.CENTER).normalize().rotate(90) * initial_speed

            self.satellites.append(Satellite(
                id=i,
                pos=pos,
                vel=vel,
                color=self.SAT_COLORS[i],
                target_speed=initial_speed
            ))

        # Initialize Debris Fields
        self.debris_fields.clear()
        self.debris_fields.append(DebrisField(
            base_radius=110, width=20, amplitude=15, 
            frequency=0.005, phase=self.np_random.uniform(0, 2*math.pi)
        ))
        self.debris_fields.append(DebrisField(
            base_radius=190, width=25, amplitude=20, 
            frequency=0.003, phase=self.np_random.uniform(0, 2*math.pi)
        ))
        
        # Third field, ensure it doesn't spawn on the target orbit to avoid instant game over
        if self.np_random.random() < 0.5:
            # Spawn below the target orbit
            rand_base_radius = self.np_random.uniform(80, 115)
        else:
            # Spawn above the target orbit
            rand_base_radius = self.np_random.uniform(185, 220)

        self.debris_fields.append(DebrisField(
            base_radius=rand_base_radius, width=15, amplitude=10, 
            frequency=0.008, phase=self.np_random.uniform(0, 2*math.pi)
        ))
        
        # Ensure first selected satellite is alive
        self._find_next_alive_satellite(1)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Action Handling ---
        self._handle_input(movement, space_held, shift_held)

        # --- Game Logic Update ---
        self.steps += 1
        self._update_debris_fields()
        self._update_satellites()
        self._update_particles()
        
        # --- Collision Detection and Reward Calculation ---
        reward, terminated = self._process_interactions_and_rewards()
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Cooldown for selection to prevent rapid cycling
        if self.action_cooldowns['selection'] > 0:
            self.action_cooldowns['selection'] -= 1
        
        num_alive = sum(1 for s in self.satellites if s.alive)
        if num_alive > 0 and self.action_cooldowns['selection'] == 0:
            if movement == 1: # Up -> Previous
                self._find_next_alive_satellite(-1)
                self.action_cooldowns['selection'] = 10 # 10 frames cooldown
            elif movement == 2: # Down -> Next
                self._find_next_alive_satellite(1)
                self.action_cooldowns['selection'] = 10

        # Adjust speed of selected satellite
        if num_alive > 0 and self.satellites[self.selected_satellite_idx].alive:
            sat = self.satellites[self.selected_satellite_idx]
            if space_held: # Increase speed
                sat.target_speed = min(self.MAX_SAT_SPEED, sat.target_speed + self.SPEED_ADJUST_RATE)
            if shift_held: # Decrease speed
                sat.target_speed = max(self.MIN_SAT_SPEED, sat.target_speed - self.SPEED_ADJUST_RATE)

    def _find_next_alive_satellite(self, direction):
        num_sats = len(self.satellites)
        if sum(1 for s in self.satellites if s.alive) == 0:
            return

        start_idx = self.selected_satellite_idx
        while True:
            self.selected_satellite_idx = (self.selected_satellite_idx + direction + num_sats) % num_sats
            if self.satellites[self.selected_satellite_idx].alive:
                break
            if self.selected_satellite_idx == start_idx:
                break

    def _update_debris_fields(self):
        for field in self.debris_fields:
            field.current_radius = field.base_radius + field.amplitude * math.sin(field.frequency * self.steps + field.phase)

    def _update_satellites(self):
        for sat in self.satellites:
            if not sat.alive:
                continue

            # Gravitational pull towards the center
            to_center = self.CENTER - sat.pos
            dist_sq = to_center.length_squared()
            if dist_sq > 1: # Avoid division by zero
                gravity_force = to_center.normalize() * (self.G_CONST / dist_sq)
                sat.vel += gravity_force

            # Thrust to match target speed
            current_speed = sat.vel.length()
            if current_speed > 0.01:
                speed_diff = sat.target_speed - current_speed
                correction_factor = 0.02 
                thrust_force = sat.vel.normalize() * speed_diff * correction_factor
                sat.vel += thrust_force
                
                if abs(speed_diff) > 0.1:
                    self._create_thruster_particles(sat, speed_diff)

            # Update position
            sat.pos += sat.vel
            
            # Boundary check (bounce off screen edges)
            if sat.pos.x < sat.radius or sat.pos.x > self.SCREEN_WIDTH - sat.radius:
                sat.vel.x *= -0.8
                sat.pos.x = np.clip(sat.pos.x, sat.radius, self.SCREEN_WIDTH - sat.radius)
            if sat.pos.y < sat.radius or sat.pos.y > self.SCREEN_HEIGHT - sat.radius:
                sat.vel.y *= -0.8
                sat.pos.y = np.clip(sat.pos.y, sat.radius, self.SCREEN_HEIGHT - sat.radius)

    def _process_interactions_and_rewards(self):
        reward = 0.0
        
        # Check for collisions and calculate survival reward
        for sat in self.satellites:
            if not sat.alive:
                continue
            
            dist_from_center = (sat.pos - self.CENTER).length()
            if abs(dist_from_center - self.TARGET_ORBIT_RADIUS) < self.ORBIT_RADIUS_TOLERANCE:
                reward += 0.01

            # Collision with debris
            for field in self.debris_fields:
                if abs(dist_from_center - field.current_radius) < (field.width / 2 + sat.radius):
                    sat.alive = False
                    reward -= 10
                    self._create_explosion(sat.pos, sat.color)
                    break 
        
        num_alive = sum(1 for s in self.satellites if s.alive)
        
        terminated = (num_alive == 0)
        
        # Victory reward on truncation
        if self.steps >= self.MAX_STEPS and num_alive > 0:
            reward += 100

        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_star()
        self._render_target_orbit()
        self._render_debris_fields()
        self._render_particles()
        self._render_satellites()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_star(self):
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 25, self.COLOR_STAR_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), 20, self.COLOR_STAR)

    def _render_target_orbit(self):
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.TARGET_ORBIT_RADIUS, self.COLOR_TARGET_ORBIT)

    def _render_debris_fields(self):
        for field in self.debris_fields:
            num_particles = int(field.width * 5)
            for _ in range(num_particles):
                angle = self.np_random.uniform(0, 2 * math.pi)
                radius_offset = self.np_random.uniform(-field.width / 2, field.width / 2)
                r = field.current_radius + radius_offset
                pos_x = int(self.CENTER.x + r * math.cos(angle))
                pos_y = int(self.CENTER.y + r * math.sin(angle))
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 1, self.COLOR_DEBRIS)

    def _render_satellites(self):
        num_alive = sum(1 for s in self.satellites if s.alive)
        for i, sat in enumerate(self.satellites):
            if not sat.alive:
                continue
            
            pos_int = (int(sat.pos.x), int(sat.pos.y))
            
            if i == self.selected_satellite_idx and num_alive > 0:
                pulse_rad = sat.radius + 5 + int(2 * math.sin(self.steps * 0.1))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], pulse_rad, self.COLOR_SELECTION)
            
            glow_radius = int(sat.radius * 1.5)
            glow_color = (*sat.color, 80)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, glow_color)
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], sat.radius, sat.color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], sat.radius, (255,255,255,150))
            
            speed_text = f"{sat.vel.length():.1f}"
            text_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (pos_int[0] + 12, pos_int[1] - 6))

    def _render_ui(self):
        time_left = (self.MAX_STEPS - self.steps) / self.metadata['render_fps']
        timer_text = f"TIME: {max(0, time_left):.1f}s"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        num_alive = sum(1 for s in self.satellites if s.alive)
        sat_text = f"SATELLITES: {num_alive}/{self.NUM_SATELLITES}"
        sat_surf = self.font_large.render(sat_text, True, self.COLOR_TEXT)
        text_rect = sat_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(sat_surf, text_rect)

    def _get_info(self):
        num_alive = sum(1 for s in self.satellites if s.alive)
        return {
            "score": self.score,
            "steps": self.steps,
            "satellites_alive": num_alive,
            "selected_satellite": self.selected_satellite_idx if num_alive > 0 else -1,
        }
        
    def _create_explosion(self, pos, color):
        for _ in range(50):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            lifespan = self.np_random.integers(20, 40)
            self.particles.append([pos.copy(), vel, lifespan, color])
            
    def _create_thruster_particles(self, sat, speed_diff):
        if self.steps % 3 != 0: return
        thrust_dir = -sat.vel.normalize() if speed_diff > 0 else sat.vel.normalize()
        for _ in range(2):
            vel = thrust_dir * self.np_random.uniform(1, 2) + pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            lifespan = self.np_random.integers(10, 20)
            start_pos = sat.pos + thrust_dir * sat.radius
            self.particles.append([start_pos, vel, lifespan, (255, 255, 255)])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 1]
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 1

    def _render_particles(self):
        for pos, vel, lifespan, color in self.particles:
            alpha = int(255 * (lifespan / 40.0))
            final_color = (*color[:3], max(0, min(255, alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 2, final_color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will use a real screen, not the dummy one.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Orbital Debris")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    done = False
    total_reward = 0
    
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

    print(f"Episode finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()