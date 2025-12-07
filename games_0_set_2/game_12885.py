import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:24:04.380273
# Source Brief: brief_02885.md
# Brief Index: 2885
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a survival game.
    The player must absorb energy orbs to create shields and survive three
    waves of enemy projectiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of enemy projectiles by moving your ship to collect energy orbs. "
        "Use the collected energy to create shields and outlast the onslaught."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move. "
        "Press space to spend energy and create a shield."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 5
    ORB_RADIUS_MIN = 5
    ORB_RADIUS_MAX = 10
    MAX_ORBS = 15
    ENERGY_PER_SHIELD = 100
    MAX_ENERGY = 1000
    MAX_SHIELDS = 8
    INITIAL_SHIELDS = 3
    MAX_STEPS = 1500 # Increased from 1000 to allow more time for 3 waves
    WAVE_PROJECTILES = {1: 10, 2: 15, 3: 20} # Increased projectile counts for more challenge
    WAVE_INTERVAL = 90  # 3 seconds at 30fps

    # --- COLORS ---
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_ORB = (50, 150, 255)
    COLOR_PROJECTILE = (255, 50, 50)
    COLOR_SHIELD = (50, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE_ORB = (100, 200, 255)
    COLOR_PARTICLE_HIT = (255, 150, 50)

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
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_wave = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_energy = None
        self.player_shields = None
        self.current_wave = None
        self.wave_timer = None
        self.projectiles = None
        self.orbs = None
        self.particles = None
        self.steps = None
        self.score = None
        self.prev_space_held = None
        self.game_over = None
        self.win_condition_met = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.player_energy = 0
        self.player_shields = self.INITIAL_SHIELDS
        self.current_wave = 0
        self.wave_timer = self.WAVE_INTERVAL
        
        self.projectiles = []
        self.orbs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.prev_space_held = False
        self.game_over = False
        self.win_condition_met = False

        # Pre-populate with orbs
        for _ in range(self.MAX_ORBS):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        self._handle_input(action)
        self._update_game_state()

        # --- REWARD CALCULATION ---
        # Reward is calculated inside helper methods (_handle_collisions)
        # and _handle_input for shield creation.
        reward += self.step_reward
        self.score += self.step_reward

        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.win_condition_met:
                reward += 100
                self.score += 100
            elif self.player_shields <= 0:
                reward -= 100
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # --- Shield Creation ---
        if space_held and not self.prev_space_held:
            if self.player_energy >= self.ENERGY_PER_SHIELD and self.player_shields < self.MAX_SHIELDS:
                self.player_energy -= self.ENERGY_PER_SHIELD
                self.player_shields += 1
                self.step_reward += 1.0
                # sfx: SHIELD_UP
                self._create_particles(self.player_pos, 20, self.COLOR_SHIELD, 2, 4, 30)


        self.prev_space_held = space_held
        
    def _update_game_state(self):
        self.step_reward = 0.0 # Reset per-step reward

        self._update_projectiles()
        self._update_orbs()
        self._update_particles()
        self._update_wave_system()
        self._handle_collisions()

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not self.screen.get_rect().collidepoint(p['pos']):
                self.projectiles.remove(p)

    def _update_orbs(self):
        if len(self.orbs) < self.MAX_ORBS and self.np_random.random() < 0.1:
            self._spawn_orb()

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_wave_system(self):
        if not self.projectiles and self.current_wave in self.WAVE_PROJECTILES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self._start_wave(self.current_wave)
                self.wave_timer = self.WAVE_INTERVAL

    def _handle_collisions(self):
        # Player vs Orbs
        for orb in self.orbs[:]:
            if self.player_pos.distance_to(orb['pos']) < self.PLAYER_RADIUS + orb['radius']:
                self.player_energy = min(self.MAX_ENERGY, self.player_energy + orb['value'])
                self.step_reward += 0.1 * orb['value']
                self.orbs.remove(orb)
                # sfx: ORB_COLLECT
                self._create_particles(orb['pos'], 15, self.COLOR_PARTICLE_ORB, 1, 3, 20)

        # Player vs Projectiles
        for p in self.projectiles[:]:
            if self.player_pos.distance_to(p['pos']) < self.PLAYER_RADIUS:
                self.projectiles.remove(p)
                self.player_shields -= 1
                # sfx: SHIELD_HIT
                self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_HIT, 2, 5, 40)
                if self.player_shields <= 0:
                    self.game_over = True
                    # sfx: PLAYER_DEATH

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Win condition: Survived all waves
        if self.current_wave > len(self.WAVE_PROJECTILES) and not self.projectiles:
            if self.player_shields >= 2:
                self.win_condition_met = True
            return True
            
        return False

    def _start_wave(self, wave_num):
        num_projectiles = self.WAVE_PROJECTILES.get(wave_num, 0)
        speed = 1.0 + (wave_num - 1) * 1.0 # Increased speed scaling

        for _ in range(num_projectiles):
            # Spawn projectile from a random edge
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -10)
            elif edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
            elif edge == 2: # Left
                pos = pygame.math.Vector2(-10, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            # Aim towards the center of the screen with some variance
            target = pygame.math.Vector2(
                self.SCREEN_WIDTH/2 + self.np_random.uniform(-100, 100),
                self.SCREEN_HEIGHT/2 + self.np_random.uniform(-100, 100)
            )
            vel = (target - pos).normalize() * speed
            self.projectiles.append({'pos': pos, 'vel': vel})

    def _spawn_orb(self):
        pos = pygame.math.Vector2(
            self.np_random.uniform(self.ORB_RADIUS_MAX, self.SCREEN_WIDTH - self.ORB_RADIUS_MAX),
            self.np_random.uniform(self.ORB_RADIUS_MAX, self.SCREEN_HEIGHT - self.ORB_RADIUS_MAX)
        )
        value = self.np_random.integers(10, 51) # Increased orb value
        radius = self.ORB_RADIUS_MIN + (self.ORB_RADIUS_MAX - self.ORB_RADIUS_MIN) * (value / 50)
        self.orbs.append({
            'pos': pos,
            'value': value,
            'radius': radius,
            'pulse_offset': self.np_random.uniform(0, 2 * math.pi)
        })

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan // 2, lifespan),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.player_energy,
            "shields": self.player_shields,
            "wave": self.current_wave,
            "projectiles": len(self.projectiles)
        }

    def _render_game(self):
        # Render Particles (draw first, so they are behind other objects)
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30)) if p['lifespan'] < 30 else 255
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Render Orbs
        for orb in self.orbs:
            pulse = math.sin(self.steps * 0.1 + orb['pulse_offset']) * 2
            radius = int(orb['radius'] + pulse)
            pygame.gfxdraw.filled_circle(self.screen, int(orb['pos'].x), int(orb['pos'].y), radius, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, int(orb['pos'].x), int(orb['pos'].y), radius, self.COLOR_ORB)

        # Render Projectiles
        for p in self.projectiles:
            start_pos = (int(p['pos'].x), int(p['pos'].y))
            end_pos = (int(p['pos'].x - p['vel'].x * 5), int(p['pos'].y - p['vel'].y * 5))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)

        # Render Player
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        # Glow effect
        for i in range(4):
            glow_radius = self.PLAYER_RADIUS + i * 3
            alpha = 60 - i * 15
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, (*self.COLOR_PLAYER_GLOW, alpha))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Render Shields
        for i in range(self.player_shields):
            radius = self.PLAYER_RADIUS + 8 + i * 4
            rect = pygame.Rect(px - radius, py - radius, radius * 2, radius * 2)
            # Make shields pulse slightly
            angle_offset = (math.sin(self.steps * 0.05 + i) * 0.2)
            start_angle = 0 + angle_offset
            stop_angle = 2 * math.pi + angle_offset
            pygame.draw.arc(self.screen, self.COLOR_SHIELD, rect, start_angle, stop_angle, 3)

    def _render_ui(self):
        # Energy Bar
        energy_text = self.font_main.render(f"ENERGY: {self.player_energy}/{self.MAX_ENERGY}", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (10, 10))

        # Wave Text
        wave_str = f"WAVE: {self.current_wave}" if self.current_wave > 0 else "GET READY"
        wave_text = self.font_main.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Shield Icons
        shield_text = self.font_main.render("SHIELDS:", True, self.COLOR_TEXT)
        self.screen.blit(shield_text, (10, 40))
        for i in range(self.player_shields):
            pygame.gfxdraw.filled_circle(self.screen, 120 + i * 20, 52, 6, self.COLOR_SHIELD)
            pygame.gfxdraw.aacircle(self.screen, 120 + i * 20, 52, 6, self.COLOR_SHIELD)


        # Wave transition text
        if not self.projectiles and self.current_wave in self.WAVE_PROJECTILES:
            countdown = math.ceil(self.wave_timer / 30)
            text_str = f"WAVE {self.current_wave + 1} IN {countdown}"
            text_surf = self.font_wave.render(text_str, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
        
        # Game Over / Win Text
        if self.game_over or self.win_condition_met:
            text_str = "VICTORY" if self.win_condition_met else "GAME OVER"
            color = self.COLOR_SHIELD if self.win_condition_met else self.COLOR_PROJECTILE
            text_surf = self.font_wave.render(text_str, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for interactive testing, not part of the Gymnasium environment API
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Orb Survivor")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered frame
        # We need to transpose it back for pygame's display format (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key press
                
        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Survived to Wave: {info['wave']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()