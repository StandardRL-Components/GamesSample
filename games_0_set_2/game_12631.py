import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:10:47.925620
# Source Brief: brief_02631.md
# Brief Index: 2631
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# --- Helper Classes for Game Entities ---

class Particle:
    def __init__(self, pos, color, min_speed, max_speed, lifetime):
        self.pos = list(pos)
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            radius = int(2 * (self.lifetime / self.initial_lifetime))
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos[0]) - radius, int(self.pos[1]) - radius))

class Signal:
    def __init__(self, pos, vel, signal_type, color, speed):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.type = signal_type
        self.color = color
        self.radius = 8
        self.speed = speed
        # Normalize velocity and apply speed
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = (self.vel / norm) * self.speed
        self.trail = deque(maxlen=15)

    def update(self):
        self.trail.append(tuple(self.pos))
        self.pos += self.vel

    def draw(self, surface):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = int(100 * (i / len(self.trail)))
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), 2, (*self.color, alpha))

        # Draw glow
        for i in range(self.radius, 0, -2):
            alpha = 80 - (i * 10)
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), i, (*self.color, max(0, alpha)))
        
        # Draw main body
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)

class Redirector:
    def __init__(self, grid_pos, direction_vec, lifetime=45):
        self.grid_pos = grid_pos
        self.direction_vec = np.array(direction_vec, dtype=float)
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.lifetime -= 1

    def draw(self, surface, cell_size):
        if self.lifetime > 0:
            center_x = self.grid_pos[0] * cell_size + cell_size // 2
            center_y = self.grid_pos[1] * cell_size + cell_size // 2
            
            end_x = center_x + self.direction_vec[0] * cell_size * 0.4
            end_y = center_y + self.direction_vec[1] * cell_size * 0.4
            
            alpha = int(255 * math.sin(math.pi * (self.lifetime / self.initial_lifetime)))
            color = (100, 200, 255, alpha)
            
            # Draw multiple lines for a glow effect
            pygame.draw.line(surface, color, (center_x, center_y), (end_x, end_y), 5)
            pygame.draw.line(surface, (200, 220, 255, alpha // 2), (center_x, center_y), (end_x, end_y), 9)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your central core from incoming hostile signals by placing directional fields to redirect them."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to place redirectors. Hold space to expand your core and shift to shrink it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 20
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_CORE = (0, 255, 255)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_GREEN = (50, 255, 50)
        self.COLOR_YELLOW = (255, 255, 50)
        self.COLOR_PLAYER = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 240)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.health = 0
        self.wave = 0
        self.combo_multiplier = 1.0
        self.upgrade_points = 0
        self.core_radius = 0
        self.player_grid_pos = [0, 0]
        self.signals = []
        self.redirectors = []
        self.particles = []
        self.wave_signal_count = 0
        self.reward_this_step = 0
        
        # self.reset() # Removed to avoid premature initialization before all attributes are set
        # self.validate_implementation() # Removed for production code
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.health = 100.0
        self.wave = 1
        self.combo_multiplier = 1.0
        self.upgrade_points = 0
        
        self.core_radius = 40.0
        self.player_grid_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.signals.clear()
        self.redirectors.clear()
        self.particles.clear()
        
        self._start_wave()
        
        return self._get_observation(), self._get_info()

    def _start_wave(self):
        self.wave_signal_count = 5 + self.wave
        signal_speed = 1.5 + (self.wave * 0.05)
        
        for _ in range(self.wave_signal_count):
            self._spawn_signal('red', signal_speed)
        
        if self.wave > 5:
            for _ in range(random.randint(1, 2)):
                self._spawn_signal('green', signal_speed * 0.8)
        
        if self.wave > 10:
            for _ in range(random.randint(0, 1)):
                self._spawn_signal('yellow', signal_speed * 0.9)

    def _spawn_signal(self, s_type, speed):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = [random.uniform(0, self.WIDTH), -20]
        elif edge == 'bottom':
            pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 20]
        elif edge == 'left':
            pos = [-20, random.uniform(0, self.HEIGHT)]
        else: # right
            pos = [self.WIDTH + 20, random.uniform(0, self.HEIGHT)]

        target_pos = [
            self.WIDTH / 2 + random.uniform(-100, 100),
            self.HEIGHT / 2 + random.uniform(-100, 100)
        ]
        vel = [target_pos[0] - pos[0], target_pos[1] - pos[1]]

        color = self.COLOR_RED
        if s_type == 'green': color = self.COLOR_GREEN
        elif s_type == 'yellow': color = self.COLOR_YELLOW

        self.signals.append(Signal(pos, vel, s_type, color, speed))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_redirectors()
        self._update_signals()
        self._handle_signal_collisions()
        self._update_particles()
        self._check_wave_completion()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
            if self.wave > self.MAX_WAVES:
                self.reward_this_step += 100 # Win bonus
        
        return self._get_observation(), self.reward_this_step, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Grow/Shrink Core
        if space_held: self.core_radius = min(80, self.core_radius + 0.5)
        if shift_held: self.core_radius = max(20, self.core_radius - 0.5)

        # Move Cursor & Place Redirectors
        dx, dy = 0, 0
        direction_vec = None
        if movement == 1: # Up
            dy = -1
            direction_vec = [0, -1]
        elif movement == 2: # Down
            dy = 1
            direction_vec = [0, 1]
        elif movement == 3: # Left
            dx = -1
            direction_vec = [-1, 0]
        elif movement == 4: # Right
            dx = 1
            direction_vec = [1, 0]

        if dx != 0 or dy != 0:
            new_x = self.player_grid_pos[0] + dx
            new_y = self.player_grid_pos[1] + dy
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT:
                self.player_grid_pos = [new_x, new_y]
                # Place redirector at the new position
                self.redirectors.append(Redirector(self.player_grid_pos, direction_vec))
                self.reward_this_step += 0.1 # Reward for placing a redirector
                # Sound: "blip_redirect.wav"

    def _update_redirectors(self):
        for r in self.redirectors:
            r.update()
        self.redirectors = [r for r in self.redirectors if r.lifetime > 0]

    def _update_signals(self):
        core_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        signals_to_remove = []

        for i, signal in enumerate(self.signals):
            signal.update()
            
            # Check for redirector collision
            signal_grid_pos = (int(signal.pos[0] / self.CELL_SIZE), int(signal.pos[1] / self.CELL_SIZE))
            for r in self.redirectors:
                if r.grid_pos[0] == signal_grid_pos[0] and r.grid_pos[1] == signal_grid_pos[1]:
                    new_vel = r.direction_vec * signal.speed
                    if not np.array_equal(signal.vel, new_vel):
                        signal.vel = new_vel
                        break

            # Check for core collision
            dist_to_core = np.linalg.norm(signal.pos - core_pos)
            if dist_to_core < self.core_radius + signal.radius:
                if i not in signals_to_remove:
                    signals_to_remove.append(i)
                    self._handle_core_hit(signal)
            
            # Check for passive zap
            elif signal.type == 'red' and dist_to_core < self.core_radius * 1.5 + signal.radius:
                if random.random() < 0.02 * (self.core_radius / 40): # Zap chance increases with size
                    if i not in signals_to_remove:
                        signals_to_remove.append(i)
                        self._create_particles(signal.pos, self.COLOR_CORE, 20)
                        self.score += int(1 * self.combo_multiplier)
                        self.reward_this_step += 1.0
                        # Sound: "zap.wav"
            
            # Boundary bounce
            if not (-25 < signal.pos[0] < self.WIDTH + 25 and -25 < signal.pos[1] < self.HEIGHT + 25):
                if i not in signals_to_remove:
                    signals_to_remove.append(i) # Remove if it flies too far offscreen

        if signals_to_remove:
            self.signals = [s for i, s in enumerate(self.signals) if i not in signals_to_remove]

    def _handle_core_hit(self, signal):
        # Sound: "core_hit.wav"
        if signal.type == 'red':
            damage = 10 / (self.combo_multiplier or 1)
            self.health = max(0, self.health - damage)
            self.combo_multiplier = 1.0
            self.reward_this_step -= 0.1
            self._create_particles(signal.pos, signal.color, 40)
        elif signal.type == 'green':
            self.health = min(100, self.health + 15)
            self.score += 5
            self.reward_this_step += 5.0
            self._create_particles(signal.pos, signal.color, 20, 1, 3)
            # Sound: "heal.wav"
        elif signal.type == 'yellow':
            self.combo_multiplier += 0.5
            self.score += 2
            self.reward_this_step += 2.0
            self._create_particles(signal.pos, signal.color, 20, 1, 3)
            # Sound: "combo_pickup.wav"

    def _handle_signal_collisions(self):
        signals_to_remove = set()
        red_signals = [(i, s) for i, s in enumerate(self.signals) if s.type == 'red' and i not in signals_to_remove]

        for i in range(len(red_signals)):
            for j in range(i + 1, len(red_signals)):
                idx1, sig1 = red_signals[i]
                idx2, sig2 = red_signals[j]

                if idx1 in signals_to_remove or idx2 in signals_to_remove:
                    continue

                dist = np.linalg.norm(sig1.pos - sig2.pos)
                if dist < sig1.radius + sig2.radius:
                    signals_to_remove.add(idx1)
                    signals_to_remove.add(idx2)
                    
                    mid_point = (sig1.pos + sig2.pos) / 2
                    self._create_particles(mid_point, self.COLOR_RED, 30)
                    
                    self.score += int(2 * self.combo_multiplier)
                    self.reward_this_step += 1.0  # Base reward for one elimination
                    # Sound: "signal_destroy.wav"

        if signals_to_remove:
            self.signals = [s for i, s in enumerate(self.signals) if i not in signals_to_remove]

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]
    
    def _create_particles(self, pos, color, count, min_speed=1, max_speed=4, lifetime=30):
        for _ in range(count):
            self.particles.append(Particle(pos, color, min_speed, max_speed, lifetime))

    def _check_wave_completion(self):
        if not any(s.type == 'red' for s in self.signals):
            self.wave += 1
            self.score += 50
            self.reward_this_step += 50
            self.upgrade_points += 1
            if self.wave <= self.MAX_WAVES:
                self._start_wave()
                # Sound: "wave_complete.wav"

    def _check_termination(self):
        return self.health <= 0 or self.wave > self.MAX_WAVES

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw redirectors
        for r in self.redirectors:
            r.draw(self.screen, self.CELL_SIZE)

        # Draw core
        core_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 10
        glow_radius = int(self.core_radius + pulse)
        for i in range(glow_radius, int(self.core_radius), -2):
            alpha = int(80 * ((glow_radius - i) / (glow_radius - self.core_radius)))
            pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], i, (*self.COLOR_CORE, alpha))
        pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], int(self.core_radius), self.COLOR_CORE)
        pygame.gfxdraw.filled_circle(self.screen, core_pos[0], core_pos[1], int(self.core_radius), self.COLOR_CORE)
        
        # Draw passive zap range
        zap_radius = int(self.core_radius * 1.5)
        pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], zap_radius, (*self.COLOR_CORE, 30))

        # Draw signals
        for s in self.signals:
            s.draw(self.screen)
            
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw player cursor
        px = self.player_grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.player_grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (px - 8, py), (px + 8, py), 2)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (px, py - 8), (px, py + 8), 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.health / 100)
        health_color = (int(50 + (1-health_ratio)*205), int(50 + health_ratio*205), 50)
        pygame.draw.rect(self.screen, (30,30,30), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, int(200 * health_ratio), 20))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Wave
        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 35))

        # Combo Multiplier
        if self.combo_multiplier > 1.0:
            combo_text = self.font_small.render(f"{self.combo_multiplier:.1f}x", True, self.COLOR_YELLOW)
            self.screen.blit(combo_text, (self.WIDTH // 2 - combo_text.get_width() // 2, self.HEIGHT - 55))
        
        # Upgrade Points
        up_text = self.font_small.render(f"UPGRADES: {self.upgrade_points}", True, self.COLOR_TEXT)
        self.screen.blit(up_text, (self.WIDTH - up_text.get_width() - 15, self.HEIGHT - 25))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 15, 30, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.wave > self.MAX_WAVES else "NETWORK OFFLINE"
            end_text = self.font_main.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))


    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.health, "wave": self.wave}
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Synaptic Siege")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(30) # Limit to 30 FPS for smooth manual play
        
    env.close()