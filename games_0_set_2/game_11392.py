import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:54:09.238256
# Source Brief: brief_01392.md
# Brief Index: 1392
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
        "Defend your planet from incoming solar flares by launching intercepting stardust projectiles. "
        "Survive until the timer runs out to win!"
    )
    user_guide = (
        "Controls: Use ↑/↓ to aim the launcher and ←/→ to adjust power. Press Space to fire and Shift to cycle ammo types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    VICTORY_STEPS = 2000
    LAUNCH_ORIGIN = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)

    # --- Colors ---
    COLOR_BG = (10, 5, 25)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLANET = (60, 90, 180)
    COLOR_PLANET_ATMOSPHERE = (100, 130, 220, 50)
    COLOR_FLARE = (255, 80, 20)
    COLOR_FLARE_GLOW = (255, 120, 60, 80)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_TRAJECTORY = (255, 255, 255, 100)
    STARDUST_COLORS = {
        1: (80, 150, 255),
        2: (80, 255, 150),
        3: (255, 220, 80)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # --- State Initialization ---
        self.planets = []
        self.flares = []
        self.stardust_projectiles = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.selected_stardust_type = 0
        self.unlocked_stardust_types = []
        
        self.flare_spawn_timer = 0
        self.flare_spawn_rate = 0.0
        self.flare_speed_multiplier = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here.
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # This should be run externally, not in __init__


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # --- Player State ---
        self.launch_angle = -45.0  # degrees
        self.launch_power = 50.0  # percentage
        self.selected_stardust_type = 1
        self.unlocked_stardust_types = [1]
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Entity State ---
        self.planets = [{'pos': (100, 100), 'radius': 30, 'health': 1}]
        self.flares = []
        self.stardust_projectiles = []
        self.particles = []

        # --- Difficulty State ---
        self.flare_spawn_timer = 200
        self.flare_spawn_rate = 1 / 200  # Flares per step
        self.flare_speed_multiplier = 0.2

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- 1. Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- 2. Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- 3. Update Game Logic ---
        reward = 0.1  # Survival reward
        self.steps += 1

        self._update_difficulty()
        self._update_flares()
        self._update_stardust()
        self._update_particles()
        
        # --- 4. Collision Detection and Rewards ---
        reward += self._handle_collisions()
        
        # --- 5. Check Termination and Final Rewards ---
        terminated = self._check_termination()
        if terminated:
            if self.game_over: # Planet was hit
                reward = -100.0
            elif self.steps >= self.VICTORY_STEPS: # Survived
                reward = 100.0
                self.score += 100

        truncated = self.steps >= self.VICTORY_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Adjust angle
        # movement 1 is up, 2 is down in the manual controls
        if movement == 1: self.launch_angle -= 1.5
        elif movement == 2: self.launch_angle += 1.5
        self.launch_angle = np.clip(self.launch_angle, -160, -20)

        # Adjust power
        # movement 3 is left, 4 is right in the manual controls
        if movement == 3: self.launch_power -= 1.5
        elif movement == 4: self.launch_power += 1.5
        self.launch_power = np.clip(self.launch_power, 20, 100)

        # Launch stardust (on button press)
        if space_held and not self.prev_space_held:
            self._launch_stardust()
            # sfx: launch_sound()

        # Cycle stardust type (on button press)
        if shift_held and not self.prev_shift_held and len(self.unlocked_stardust_types) > 1:
            current_index = self.unlocked_stardust_types.index(self.selected_stardust_type)
            next_index = (current_index + 1) % len(self.unlocked_stardust_types)
            self.selected_stardust_type = self.unlocked_stardust_types[next_index]
            # sfx: cycle_weapon_sound()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_difficulty(self):
        # Unlock new stardust types
        if self.steps == 500 and 2 not in self.unlocked_stardust_types:
            self.unlocked_stardust_types.append(2)
        if self.steps == 1000 and 3 not in self.unlocked_stardust_types:
            self.unlocked_stardust_types.append(3)

        # Increase flare frequency and speed every 200 steps
        if self.steps > 0 and self.steps % 200 == 0:
            self.flare_spawn_rate = min(0.1, self.flare_spawn_rate + 0.005)
            self.flare_speed_multiplier = min(1.0, self.flare_speed_multiplier + 0.005)

    def _update_flares(self):
        # Spawn new flares
        if self.np_random.random() < self.flare_spawn_rate:
            self._spawn_flare()

        # Move existing flares
        for flare in self.flares:
            flare['pos'] = (flare['pos'][0] + flare['vel'][0], flare['pos'][1] + flare['vel'][1])

    def _update_stardust(self):
        for p in self.stardust_projectiles:
            p['vel'] = (p['vel'][0], p['vel'][1] + 0.05) # Gravity
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['trail'].append(p['pos'])
            if len(p['trail']) > 15:
                p['trail'].pop(0)

        # Remove off-screen stardust
        self.stardust_projectiles = [p for p in self.stardust_projectiles if 0 < p['pos'][0] < self.SCREEN_WIDTH and p['pos'][1] < self.SCREEN_HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Stardust vs Flares
        projectiles_to_remove = set()
        flares_to_remove = set()
        
        for i, sd in enumerate(self.stardust_projectiles):
            for j, flare in enumerate(self.flares):
                dist = math.hypot(sd['pos'][0] - flare['pos'][0], sd['pos'][1] - flare['pos'][1])
                if dist < sd['radius'] + flare['radius']:
                    projectiles_to_remove.add(i)
                    flares_to_remove.add(j)
                    self._trigger_chain_reaction(sd, flare)
                    reward += 1.0
                    self.score += 1

        # Flares vs Planets (check after chain reactions)
        flares_after_reaction = [f for k, f in enumerate(self.flares) if k not in flares_to_remove]
        for flare in flares_after_reaction:
            for planet in self.planets:
                dist = math.hypot(flare['pos'][0] - planet['pos'][0], flare['pos'][1] - planet['pos'][1])
                if dist < flare['radius'] + planet['radius']:
                    self.game_over = True
                    planet['health'] = 0
                    self._create_explosion(planet['pos'], 100, (255, 200, 200), 150)
                    # sfx: planet_destruction_sound()
                    break
            if self.game_over:
                break

        # Filter out collided entities
        self.stardust_projectiles = [p for i, p in enumerate(self.stardust_projectiles) if i not in projectiles_to_remove]
        self.flares = [f for i, f in enumerate(self.flares) if i not in flares_to_remove]
        
        return reward

    def _trigger_chain_reaction(self, stardust, flare):
        # sfx: explosion_sound()
        self._create_explosion(flare['pos'], 20, self.STARDUST_COLORS[stardust['type']], 40)

        if stardust['type'] == 2: # Larger explosion
            for other_flare in self.flares:
                if other_flare is not flare:
                    dist = math.hypot(flare['pos'][0] - other_flare['pos'][0], flare['pos'][1] - other_flare['pos'][1])
                    if dist < 80: # Chain reaction radius
                        self.flares.remove(other_flare)
                        self._create_explosion(other_flare['pos'], 15, self.STARDUST_COLORS[stardust['type']], 30)
                        self.score += 1

        elif stardust['type'] == 3: # Splits into smaller projectiles
            for _ in range(3):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 2)
                new_vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                new_sd = {
                    'pos': flare['pos'], 'vel': new_vel, 'type': 1, 'radius': 3, 'trail': []
                }
                self.stardust_projectiles.append(new_sd)


    def _check_termination(self):
        return self.game_over

    def _get_observation(self):
        # --- 1. Clear Screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- 2. Render Game Elements ---
        self._render_game()
        
        # --- 3. Render UI Overlay ---
        self._render_ui()
        
        # --- 4. Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Spawning Methods ---
    def _generate_stars(self):
        self.stars = []
        # Use a fixed seed for the star background for consistency
        rng = np.random.default_rng(seed=42)
        for _ in range(150):
            self.stars.append({
                'pos': (rng.integers(0, self.SCREEN_WIDTH), rng.integers(0, self.SCREEN_HEIGHT)),
                'size': rng.uniform(0.5, 1.5)
            })

    def _spawn_flare(self):
        side = self.np_random.choice(['top', 'right'])
        if side == 'top':
            pos = (self.np_random.uniform(0, self.SCREEN_WIDTH), -10)
        else:
            pos = (self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT * 0.7))
        
        target_planet = self.np_random.choice(self.planets)
        angle = math.atan2(target_planet['pos'][1] - pos[1], target_planet['pos'][0] - pos[0])
        speed = self.np_random.uniform(0.5, 1.5) * self.flare_speed_multiplier
        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        
        self.flares.append({'pos': pos, 'vel': vel, 'radius': 6, 'spawn_time': self.steps})

    def _launch_stardust(self):
        angle_rad = math.radians(self.launch_angle)
        power_scaled = (self.launch_power / 100.0) * 10.0
        vel = (math.cos(angle_rad) * power_scaled, math.sin(angle_rad) * power_scaled)
        
        self.stardust_projectiles.append({
            'pos': self.LAUNCH_ORIGIN,
            'vel': vel,
            'type': self.selected_stardust_type,
            'radius': 5,
            'trail': []
        })

    def _create_explosion(self, pos, num_particles, color, max_speed):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed / 20.0)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos,
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    # --- Rendering Methods ---
    def _render_game(self):
        # Stars
        for star in self.stars:
            size = int(star['size'])
            pos = (int(star['pos'][0]), int(star['pos'][1]))
            if size > 0:
                pygame.draw.rect(self.screen, self.COLOR_STAR, (*pos, size, size))

        # Planets
        for planet in self.planets:
            pos = (int(planet['pos'][0]), int(planet['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos, planet['radius'] + 5, self.COLOR_PLANET_ATMOSPHERE)
            pygame.gfxdraw.aacircle(self.screen, *pos, planet['radius'] + 5, self.COLOR_PLANET_ATMOSPHERE)
            pygame.gfxdraw.filled_circle(self.screen, *pos, planet['radius'], self.COLOR_PLANET)
            pygame.gfxdraw.aacircle(self.screen, *pos, planet['radius'], self.COLOR_PLANET)

        # Flares
        for flare in self.flares:
            pos = (int(flare['pos'][0]), int(flare['pos'][1]))
            rad = int(flare['radius'])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, *pos, rad + 5, self.COLOR_FLARE_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, *pos, rad, self.COLOR_FLARE)

        # Stardust
        for p in self.stardust_projectiles:
            # Trail
            if len(p['trail']) > 1:
                trail_points = [(int(pos[0]), int(pos[1])) for pos in p['trail']]
                for i in range(len(trail_points) - 1):
                    alpha = int(255 * (i / len(p['trail'])))
                    color = (*self.STARDUST_COLORS[p['type']], alpha)
                    # This is not efficient, but gfxdraw doesn't have alpha lines
                    # For a quick fix, let's just draw circles for the trail
                    pygame.gfxdraw.filled_circle(self.screen, trail_points[i][0], trail_points[i][1], 2, color)

            # Projectile
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos, p['radius'], self.STARDUST_COLORS[p['type']])
            pygame.gfxdraw.aacircle(self.screen, *pos, p['radius'], self.STARDUST_COLORS[p['type']])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            rad = int(p['size'] * (p['lifetime'] / 40.0))
            if rad > 0:
                pygame.gfxdraw.filled_circle(self.screen, *pos, rad, color)

    def _render_ui(self):
        # --- Launch Trajectory Preview ---
        if not self.game_over:
            path = []
            angle_rad = math.radians(self.launch_angle)
            power_scaled = (self.launch_power / 100.0) * 10.0
            pos = list(self.LAUNCH_ORIGIN)
            vel = [math.cos(angle_rad) * power_scaled, math.sin(angle_rad) * power_scaled]
            for i in range(30):
                vel[1] += 0.05
                pos[0] += vel[0]
                pos[1] += vel[1]
                if i % 3 == 0:
                    path.append((int(pos[0]), int(pos[1])))
            if len(path) > 1:
                pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, path, 1)

        # --- Score and Time ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.VICTORY_STEPS - self.steps)
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # --- Launcher UI ---
        # Power Bar
        power_bar_width = 100
        power_fill = int((self.launch_power / 100.0) * power_bar_width)
        pygame.draw.rect(self.screen, (50, 50, 80), (self.LAUNCH_ORIGIN[0] - 50, self.SCREEN_HEIGHT - 20, power_bar_width, 10))
        pygame.draw.rect(self.screen, (150, 150, 255), (self.LAUNCH_ORIGIN[0] - 50, self.SCREEN_HEIGHT - 20, power_fill, 10))
        
        # Stardust Type Indicator
        icon_x = self.LAUNCH_ORIGIN[0] + 70
        for i, stype in enumerate(self.unlocked_stardust_types):
            y_pos = self.SCREEN_HEIGHT - 25 - (i * 20)
            color = self.STARDUST_COLORS[stype]
            pygame.gfxdraw.filled_circle(self.screen, icon_x, y_pos, 8, color)
            if stype == self.selected_stardust_type:
                pygame.gfxdraw.aacircle(self.screen, icon_x, y_pos, 10, self.COLOR_UI_TEXT)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Stardust Defender")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0.0
    
    while not terminated and not truncated:
        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()