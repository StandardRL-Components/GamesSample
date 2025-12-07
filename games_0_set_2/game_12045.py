import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Helper Classes ---

class Fragment:
    """Represents a single, unattached asteroid fragment."""
    def __init__(self, frag_type, pos, screen_dims, rng: np.random.Generator):
        self.type = frag_type
        self.pos = np.array(pos, dtype=float)
        self.vel = rng.uniform(-1, 1, size=2).astype(float)
        self.angle = rng.uniform(0, 360)
        self.angular_vel = rng.uniform(-1, 1)
        self.mass = 1.0
        self.radius = 15
        
        if self.type == 'hull':
            self.color = (160, 160, 170)
            self.mass = 1.2
            self.radius = 18
        elif self.type == 'engine':
            self.color = (60, 120, 220)
            self.mass = 1.5
            self.radius = 16
        elif self.type == 'weapon':
            self.color = (220, 80, 80)
            self.mass = 0.8
            self.radius = 14
        
        self.screen_width, self.screen_height = screen_dims

    def update(self, force, dt=1/30):
        # Update velocity and position
        self.vel += force / self.mass * dt
        self.pos += self.vel * dt
        
        # Apply damping
        self.vel *= 0.98
        self.angular_vel *= 0.99
        
        # Update angle
        self.angle = (self.angle + self.angular_vel * dt * 50) % 360
        
        # Boundary collision
        if self.pos[0] < self.radius or self.pos[0] > self.screen_width - self.radius:
            self.vel[0] *= -0.8
            self.pos[0] = np.clip(self.pos[0], self.radius, self.screen_width - self.radius)
        if self.pos[1] < self.radius or self.pos[1] > self.screen_height - self.radius:
            self.vel[1] *= -0.8
            self.pos[1] = np.clip(self.pos[1], self.radius, self.screen_height - self.radius)

    def draw(self, surface):
        points = []
        if self.type == 'hull': # Hexagon
            for i in range(6):
                angle_rad = math.radians(self.angle + 60 * i)
                points.append((self.pos[0] + self.radius * math.cos(angle_rad), self.pos[1] + self.radius * math.sin(angle_rad)))
        elif self.type == 'engine': # Trapezoid
            for i, offset in enumerate([(-1, -0.8), (1, -0.8), (0.6, 0.8), (-0.6, 0.8)]):
                angle_rad = math.radians(self.angle)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                x, y = offset[0] * self.radius, offset[1] * self.radius
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a
                points.append((self.pos[0] + rx, self.pos[1] + ry))
        elif self.type == 'weapon': # Triangle
            for i in range(3):
                angle_rad = math.radians(self.angle + 120 * i)
                points.append((self.pos[0] + self.radius * 1.2 * math.cos(angle_rad), self.pos[1] + self.radius * 1.2 * math.sin(angle_rad)))
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, self.color)
        pygame.gfxdraw.filled_polygon(surface, int_points, self.color)

class Ship:
    """Represents the assembled ship, composed of multiple fragments."""
    def __init__(self, fragments, screen_dims):
        self.fragments = fragments
        self.screen_width, self.screen_height = screen_dims
        self._recalculate_properties()

    def _recalculate_properties(self):
        self.mass = sum(f.mass for f in self.fragments)
        
        # Calculate center of mass and velocity
        com_pos = np.sum([f.pos * f.mass for f in self.fragments], axis=0) / self.mass
        com_vel = np.sum([f.vel * f.mass for f in self.fragments], axis=0) / self.mass
        
        self.pos = com_pos
        self.vel = com_vel
        self.angle = 0 # Ships don't rotate in construction
        self.angular_vel = 0
        
        # Recalculate fragment offsets from the new COM
        for f in self.fragments:
            f.offset = f.pos - self.pos

    def get_composition(self):
        composition = {'hull': 0, 'engine': 0, 'weapon': 0}
        for f in self.fragments:
            composition[f.type] += 1
        return composition

    def add_fragment(self, fragment):
        self.fragments.append(fragment)
        self._recalculate_properties()
        
    def update(self, force, dt=1/30):
        self.vel += force / self.mass * dt
        self.pos += self.vel * dt
        self.vel *= 0.98

        # Keep ship parts together
        for f in self.fragments:
            f.pos = self.pos + f.offset
            f.vel = self.vel
        
        # Boundary collision for the whole ship
        min_x = min(f.pos[0] - f.radius for f in self.fragments)
        max_x = max(f.pos[0] + f.radius for f in self.fragments)
        min_y = min(f.pos[1] - f.radius for f in self.fragments)
        max_y = max(f.pos[1] + f.radius for f in self.fragments)

        if min_x < 0 or max_x > self.screen_width:
            self.vel[0] *= -0.8
            # Nudge the ship back in bounds
            if min_x < 0: self.pos[0] += -min_x
            if max_x > self.screen_width: self.pos[0] -= (max_x - self.screen_width)

        if min_y < 0 or max_y > self.screen_height:
            self.vel[1] *= -0.8
            if min_y < 0: self.pos[1] += -min_y
            if max_y > self.screen_height: self.pos[1] -= (max_y - self.screen_height)

    def draw(self, surface):
        for f in self.fragments:
            f.draw(surface)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius *= 0.97
        return self.lifetime > 0 and self.radius > 0.5

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            current_color = (*self.color, alpha)
            # This is a simplified way to draw with alpha, not perfect but works
            temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, current_color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Use a magnetic tractor beam to assemble a spaceship from floating fragments. "
        "Match the mission objective, then launch to test your creation!"
    )
    user_guide = (
        "Controls: Use arrow keys to move the tractor beam. Hold space to activate a magnetic pulse. "
        "Press shift to launch and test your assembled ship."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors & Constants
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_ATTRACTOR = (255, 150, 0)
        self.ATTRACTOR_SPEED = 6.0
        self.MAGNETIC_STRENGTH = 15000.0
        self.PULSE_STRENGTH_MULT = 5.0
        self.PULSE_DURATION = 6 # steps
        self.CONNECTION_DISTANCE = 45
        self.MAX_STEPS = 1000
        
        self.stars = []
        
        self.mission_level = 0
        self.total_score = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.game_over_timer = 0
        
        self.mode = 'construction' # 'construction' or 'flight'
        self.last_shift_held = False
        
        self.attractor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.pulse_timer = 0
        
        self.fragments = []
        self.ship = None
        self.particles = []
        
        self.stars = [(
            self.np_random.integers(0, self.WIDTH, endpoint=True),
            self.np_random.integers(0, self.HEIGHT, endpoint=True),
            self.np_random.integers(1, 2, endpoint=True)
        ) for _ in range(200)]
        
        self._generate_mission()
        self._spawn_fragments()
        
        return self._get_observation(), self._get_info()

    def _generate_mission(self):
        num_components = 3 + self.mission_level // 3
        types = ['hull', 'engine', 'weapon']
        self.mission_objective = {'hull': 0, 'engine': 0, 'weapon': 0}
        for _ in range(num_components):
            self.mission_objective[self.np_random.choice(types)] += 1
            
    def _spawn_fragments(self):
        for frag_type, count in self.mission_objective.items():
            for _ in range(count):
                placed = False
                while not placed:
                    pos = self.np_random.uniform(50, [self.WIDTH - 50, self.HEIGHT - 50])
                    # Ensure fragments don't spawn on top of each other
                    if not any(np.linalg.norm(np.array(pos) - f.pos) < 50 for f in self.fragments):
                        self.fragments.append(Fragment(frag_type, pos, (self.WIDTH, self.HEIGHT), self.np_random))
                        placed = True
    
    def step(self, action):
        reward = 0.0
        terminated = False
        
        if self.game_over:
            # Allow for game over message display
            self.game_over_timer -= 1
            if self.game_over_timer <= 0:
                terminated = True
            return self._get_observation(), 0.0, terminated, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Mode Switching (Shift) ---
        # Action is edge-triggered
        if shift_held and not self.last_shift_held:
            if self.mode == 'construction':
                # Switch to flight mode to test the ship
                self.mode = 'flight'
            else: # Cannot switch back from flight mode in this design
                pass 
        self.last_shift_held = shift_held

        # --- Game Logic ---
        if self.mode == 'construction':
            reward -= 0.01 # Time penalty
            self._update_attractor(movement)
            self._update_pulse(space_held)
            
            connection_reward = self._update_physics_and_connections()
            reward += connection_reward
            
        elif self.mode == 'flight':
            # Test flight is an instantaneous event
            test_reward, success = self._run_test_flight()
            reward += test_reward
            self.game_over = True
            if success:
                self.game_over_message = "MISSION SUCCESS"
                self.game_over_timer = 60 # 2 seconds
                self.mission_level += 1
                self.total_score += self.score + test_reward # Update persistent score on success
            else:
                self.game_over_message = "ASSEMBLY FAILED"
                self.game_over_timer = 60 # 2 seconds
                self._create_explosion(self.ship.pos if self.ship else self.attractor_pos, 100)
        
        self._update_particles()
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 10.0 # Timeout penalty
            self.game_over = True
            self.game_over_message = "OUT OF TIME"
            self.game_over_timer = 60
        
        self.score += reward
        terminated = self.game_over and self.game_over_timer <= 0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_attractor(self, movement):
        if movement == 1: self.attractor_pos[1] -= self.ATTRACTOR_SPEED
        elif movement == 2: self.attractor_pos[1] += self.ATTRACTOR_SPEED
        elif movement == 3: self.attractor_pos[0] -= self.ATTRACTOR_SPEED
        elif movement == 4: self.attractor_pos[0] += self.ATTRACTOR_SPEED
        self.attractor_pos[0] = np.clip(self.attractor_pos[0], 0, self.WIDTH)
        self.attractor_pos[1] = np.clip(self.attractor_pos[1], 0, self.HEIGHT)

    def _update_pulse(self, space_held):
        if space_held and self.pulse_timer <= 0:
            self.pulse_timer = self.PULSE_DURATION
        if self.pulse_timer > 0:
            self.pulse_timer -= 1

    def _update_physics_and_connections(self):
        # Calculate forces
        entities = self.fragments + ([self.ship] if self.ship else [])
        for entity in entities:
            vec_to_attractor = self.attractor_pos - entity.pos
            dist_sq = np.dot(vec_to_attractor, vec_to_attractor) + 1e-6
            force_mag = self.MAGNETIC_STRENGTH / dist_sq
            if self.pulse_timer > 0:
                force_mag *= self.PULSE_STRENGTH_MULT
            force = (vec_to_attractor / math.sqrt(dist_sq)) * force_mag
            entity.update(force)

        # Check for connections
        connection_reward = 0
        # Fragment to Fragment
        for i in range(len(self.fragments) - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                f1 = self.fragments[i]
                f2 = self.fragments[j]
                if np.linalg.norm(f1.pos - f2.pos) < self.CONNECTION_DISTANCE:
                    if self.ship is None:
                        self.ship = Ship([f1, f2], (self.WIDTH, self.HEIGHT))
                    else: # This case should not happen if logic is correct
                        self.ship.add_fragment(f1)
                        self.ship.add_fragment(f2)
                    self._create_connection_effect((f1.pos + f2.pos) / 2)
                    self.fragments.pop(i)
                    self.fragments.pop(j)
                    connection_reward += 1.0
                    return connection_reward # Only one connection per step

        # Fragment to Ship
        if self.ship:
            for i in range(len(self.fragments) - 1, -1, -1):
                frag = self.fragments[i]
                if np.linalg.norm(frag.pos - self.ship.pos) < self.CONNECTION_DISTANCE + self.ship.mass * 5:
                    self.ship.add_fragment(frag)
                    self._create_connection_effect((frag.pos + self.ship.pos) / 2)
                    self.fragments.pop(i)
                    connection_reward += 1.0
                    return connection_reward
        return connection_reward

    def _run_test_flight(self):
        if not self.ship:
            return -5.0, False # Failed: built nothing
        
        current_composition = self.ship.get_composition()
        if current_composition == self.mission_objective:
            return 10.0, True # Success
        else:
            return -5.0, False # Failed: wrong parts

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _create_connection_effect(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 25, endpoint=True)
            self.particles.append(Particle(pos, vel, radius, (50, 255, 100), lifetime))

    def _create_explosion(self, pos, num_particles):
        colors = [(255,100,0), (255,200,0), (200,200,200)]
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(3, 8)
            lifetime = self.np_random.integers(30, 60, endpoint=True)
            color = self.np_random.choice(colors)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw stars with parallax
        for x, y, z in self.stars:
            px = (x - self.attractor_pos[0] * 0.05 * z) % self.WIDTH
            py = (y - self.attractor_pos[1] * 0.05 * z) % self.HEIGHT
            pygame.draw.circle(self.screen, (255,255,255), (int(px), int(py)), z)

        # Draw magnetic field
        if self.pulse_timer > 0:
            pulse_alpha = 1 - (self.pulse_timer / self.PULSE_DURATION)
            radius = int(20 + 80 * pulse_alpha)
            alpha = int(150 * (1 - pulse_alpha))
            if radius > 20:
                pygame.gfxdraw.aacircle(self.screen, int(self.attractor_pos[0]), int(self.attractor_pos[1]), radius, (*self.COLOR_ATTRACTOR, alpha))
        
        # Draw attractor
        pygame.gfxdraw.filled_circle(self.screen, int(self.attractor_pos[0]), int(self.attractor_pos[1]), 8, self.COLOR_ATTRACTOR)
        pygame.gfxdraw.aacircle(self.screen, int(self.attractor_pos[0]), int(self.attractor_pos[1]), 8, self.COLOR_ATTRACTOR)

        # Draw entities
        if self.ship: self.ship.draw(self.screen)
        for frag in self.fragments: frag.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)

        # Draw game over message
        if self.game_over and self.game_over_timer > 0:
            color = (100, 255, 100) if "SUCCESS" in self.game_over_message else (255, 100, 100)
            text_surf = self.font_msg.render(self.game_over_message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Mission Objective
        obj_text = "Objective: " + " ".join([f"{k[0].upper()}:{v}" for k, v in self.mission_objective.items()])
        text_surf = self.font_ui.render(obj_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Time
        time_text = f"Time: {self.MAX_STEPS - self.steps}"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Mode Icon
        icon_pos = (text_rect.left - 30, text_rect.centery)
        if self.mode == 'construction': # Wrench icon
            pygame.draw.circle(self.screen, self.COLOR_TEXT, icon_pos, 8, 2)
            pygame.draw.line(self.screen, self.COLOR_TEXT, (icon_pos[0]+5, icon_pos[1]-5), (icon_pos[0]+12, icon_pos[1]-12), 2)
        else: # Rocket icon
            pygame.gfxdraw.aapolygon(self.screen, [(icon_pos[0], icon_pos[1]-8), (icon_pos[0]+5, icon_pos[1]+8), (icon_pos[0]-5, icon_pos[1]+8)], self.COLOR_TEXT)

        # Score
        score_text = f"Score: {self.total_score + self.score:.2f}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(bottomright=(self.WIDTH - 10, self.HEIGHT - 10))
        self.screen.blit(text_surf, text_rect)

        # Current Assembly
        if self.ship:
            comp = self.ship.get_composition()
            comp_text = "Assembly: " + " ".join([f"{k[0].upper()}:{v}" for k, v in comp.items()])
            text_surf = self.font_ui.render(comp_text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (10, self.HEIGHT - 30))
            
    def _get_info(self):
        return {
            "score": self.score,
            "total_score": self.total_score,
            "steps": self.steps,
            "mission_level": self.mission_level,
            "mode": self.mode,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To display the game, you might need to remove/comment the os.environ line
    # and change the render_mode in the GameEnv constructor.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # For headless execution, this part is not strictly necessary but good for testing
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Astro-Assembler")
        has_display = True
    except pygame.error:
        has_display = False

    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released
        
        if has_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r: # Manual reset
                        obs, info = env.reset()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        else:
            # In a truly headless mode, you'd get actions from an agent
            # For this test script, we can sample random actions
            action = env.action_space.sample()
            movement, space_held, shift_held = action.tolist()

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Total Score: {info['total_score']:.2f}")
            obs, info = env.reset()
        
        if has_display:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30)
        
    env.close()