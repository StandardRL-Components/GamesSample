import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:06:54.357147
# Source Brief: brief_00273.md
# Brief Index: 273
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Organelles
class Organelle:
    def __init__(self, pos, radius, color, base_x, amplitude, frequency):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.color = color
        self.base_x = base_x
        self.amplitude = amplitude
        self.frequency = frequency
        self.base_y = pos[1]

    def update(self, steps, player_accel_x, screen_width, screen_height):
        # Oscillation based on time (steps)
        oscillation_offset = self.amplitude * math.sin(steps * self.frequency)
        
        # Player control application
        self.vel.x += player_accel_x
        self.vel *= 0.92  # Friction/Damping
        
        # Update position
        # The final position is a sum of its base position, oscillation, and player-driven velocity
        self.pos.x = self.base_x + oscillation_offset + self.vel.x
        
        # Clamp to screen boundaries
        self.pos.x = max(self.radius, min(screen_width - self.radius, self.pos.x))
        self.pos.y = self.base_y # No vertical movement

    def draw(self, surface, is_selected, tick):
        # Pulsating glow for the selected organelle
        if is_selected:
            pulse = abs(math.sin(tick * 0.005))
            glow_radius = int(self.radius * (1.4 + pulse * 0.4))
            glow_alpha = int(80 + pulse * 60)
            
            # Use a temporary surface for a nice additive glow
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.color + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
            surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw main body with anti-aliasing
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        
        # Add a brighter highlight for a 3D feel
        highlight_color = tuple(min(255, c + 60) for c in self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius * 0.7), highlight_color)

# Helper class for visual effects particles
class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.98 # Particles slow down
        self.lifetime -= 1
        self.size = max(0, self.size * 0.97)

    def draw(self, surface):
        if self.lifetime > 0 and self.size > 0.5:
            # Fade out over time
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            
            # Use a temporary surface for alpha blending
            particle_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, self.color + (alpha,), (int(self.size), int(self.size)), int(self.size))
            surface.blit(particle_surf, (int(self.pos.x - self.size), int(self.pos.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a group of oscillating organelles to absorb all the nutrients before time runs out."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the selected organelle. Press space to switch between different organelles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30 # For smooth visual interpolation
    
    # Gameplay settings
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * TARGET_FPS
    WIN_SCORE = 100
    NUM_NUTRIENTS = 120 # Spawn more than needed
    PLAYER_ACCELERATION = 1.5

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 10, 30)
    COLOR_NUTRIENT = (255, 255, 150)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (20, 20, 40)
    
    # Organelle specs: [color, radius, base_y, amplitude, frequency]
    ORGANELLE_SPECS = [
        ((255, 80, 80), 20, 100, 150, 0.02),  # Red
        ((80, 255, 80), 25, 200, 200, 0.015), # Green
        ((80, 80, 255), 18, 300, 100, 0.025), # Blue
    ]
    
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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Pre-render background for efficiency
        self.background = self._create_background()
        
        self.organelles = []
        self.nutrients = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.selected_organelle_idx = 0
        self.prev_space_held = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.selected_organelle_idx = 0
        self.prev_space_held = False
        
        self.organelles.clear()
        for spec in self.ORGANELLE_SPECS:
            color, radius, base_y, amp, freq = spec
            org = Organelle(
                pos=(self.SCREEN_WIDTH / 2, base_y),
                radius=radius,
                color=color,
                base_x=self.SCREEN_WIDTH / 2,
                amplitude=amp,
                frequency=freq
            )
            self.organelles.append(org)

        self.nutrients.clear()
        # Ensure some nutrients are initially reachable
        for org in self.organelles:
            for _ in range(5):
                angle = self.np_random.uniform(0, 2 * math.pi)
                dist = self.np_random.uniform(50, 150)
                x = org.pos.x + math.cos(angle) * dist
                y = org.pos.y + math.sin(angle) * dist
                self.nutrients.append(pygame.Vector2(x, y))
        # Add remaining nutrients randomly
        for _ in range(self.NUM_NUTRIENTS - len(self.nutrients)):
            x = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            y = self.np_random.integers(20, self.SCREEN_HEIGHT - 20)
            self.nutrients.append(pygame.Vector2(x, y))
            
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- 1. Handle Actions ---
        # Organelle switching on space press (not hold)
        if space_held and not self.prev_space_held:
            self.selected_organelle_idx = (self.selected_organelle_idx + 1) % len(self.organelles)
            # sfx: switch_organelle
        self.prev_space_held = space_held

        player_accel_x = 0
        if movement == 3:  # Left
            player_accel_x = -self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            player_accel_x = self.PLAYER_ACCELERATION
            
        # --- 2. Update Game State ---
        self.steps += 1
        self.time_remaining -= 1
        
        # Store pre-update state for reward calculation
        selected_org = self.organelles[self.selected_organelle_idx]
        dist_before = self._get_dist_to_nearest_nutrient(selected_org)

        # Update all organelles
        for i, org in enumerate(self.organelles):
            accel = player_accel_x if i == self.selected_organelle_idx else 0
            org.update(self.steps, accel, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        
        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()
            
        # --- 3. Handle Collisions & Events ---
        # Organelle-Nutrient collisions
        nutrients_absorbed = 0
        nutrients_remaining = []
        for nutrient_pos in self.nutrients:
            absorbed = False
            for org in self.organelles:
                if org.pos.distance_to(nutrient_pos) < org.radius + 5: # 5 is nutrient radius
                    self.score += 1
                    nutrients_absorbed += 1
                    absorbed = True
                    self._create_absorption_particles(nutrient_pos)
                    # sfx: absorb_nutrient
                    break
            if not absorbed:
                nutrients_remaining.append(nutrient_pos)
        self.nutrients = nutrients_remaining
        
        # Organelle-Organelle collisions
        collisions = 0
        for i in range(len(self.organelles)):
            for j in range(i + 1, len(self.organelles)):
                org1 = self.organelles[i]
                org2 = self.organelles[j]
                dist = org1.pos.distance_to(org2.pos)
                if dist < org1.radius + org2.radius:
                    collisions += 1
                    # Apply speed reduction penalty
                    org1.vel *= 0.5
                    org2.vel *= 0.5
                    # sfx: organelle_collide
                    # Create collision flash effect
                    mid_point = org1.pos.lerp(org2.pos, 0.5)
                    self._create_collision_flash(mid_point)

        # --- 4. Calculate Reward ---
        reward += nutrients_absorbed * 1.0
        reward -= collisions * 0.5
        
        # Movement reward (towards/away from nearest nutrient)
        dist_after = self._get_dist_to_nearest_nutrient(selected_org)
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.1

        # --- 5. Check Termination ---
        terminated = self.score >= self.WIN_SCORE or self.time_remaining <= 0
        truncated = False # This environment does not truncate based on step count alone

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
                # sfx: win_game
            else:
                reward -= 10.0 # Time out penalty
                # sfx: lose_game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_dist_to_nearest_nutrient(self, organelle):
        if not self.nutrients:
            return None
        return min(organelle.pos.distance_to(n) for n in self.nutrients)

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            # Linear interpolation for gradient
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        # Render particles (drawn first, underneath other elements)
        for p in self.particles:
            p.draw(self.screen)
            
        # Render nutrients
        for pos in self.nutrients:
            x, y = int(pos.x), int(pos.y)
            # Glowing effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_NUTRIENT + (30,))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 5, self.COLOR_NUTRIENT + (60,))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, self.COLOR_NUTRIENT)

        # Render organelles
        tick = pygame.time.get_ticks()
        for i, org in enumerate(self.organelles):
            org.draw(self.screen, i == self.selected_organelle_idx, tick)

    def _render_ui(self):
        # Score display
        score_text = f"NUTRIENTS: {self.score} / {self.WIN_SCORE}"
        self._draw_text(score_text, (10, 10))
        
        # Timer display
        time_left_secs = self.time_remaining / self.TARGET_FPS
        timer_text = f"TIME: {max(0, time_left_secs):.2f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 170, 10))

        # Selected organelle indicator
        indicator_text = f"CONTROLLING ORGANELLE {self.selected_organelle_idx + 1}"
        text_width = self.font.size(indicator_text)[0]
        self._draw_text(indicator_text, ((self.SCREEN_WIDTH - text_width) / 2, self.SCREEN_HEIGHT - 35))

    def _draw_text(self, text, pos):
        # Draw shadow first
        shadow_surface = self.font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        # Draw main text
        text_surface = self.font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, pos)

    def _create_absorption_particles(self, pos):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = random.uniform(2, 5)
            lifetime = random.randint(15, 25)
            self.particles.append(Particle(pos, vel, self.COLOR_NUTRIENT, size, lifetime))

    def _create_collision_flash(self, pos):
        # A single, large, quick-fading particle for a flash effect
        self.particles.append(Particle(pos, pygame.Vector2(0,0), (255,255,255), 30, 5))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Organelle Oscillator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, 0] # shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()