import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:54:17.587361
# Source Brief: brief_01353.md
# Brief Index: 1353
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes for game objects
class Particle:
    def __init__(self, pos, vel, radius, color, life):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.radius *= 0.98
        return self.life > 0 and self.radius > 0.5

class SonarPulse:
    def __init__(self, origin, max_radius, speed, color, life, chain_level=0):
        self.origin = np.array(origin, dtype=float)
        self.radius = 0.0
        self.max_radius = max_radius
        self.speed = speed
        self.color = color
        self.life = life
        self.max_life = life
        self.chain_level = chain_level
        self.hit_fish = set()

    def update(self):
        self.radius += self.speed
        self.life -= 1
        return self.life > 0 and self.radius < self.max_radius

    def get_alpha(self):
        return int(255 * (self.life / self.max_life) * (1 - self.radius / self.max_radius))

class Anglerfish:
    def __init__(self, pos, speed, lure_color, body_color):
        self.pos = np.array(pos, dtype=float)
        self.speed = speed
        self.lure_color = lure_color
        self.body_color = body_color
        self.trail = deque(maxlen=10)

    def update(self, target_pos):
        self.trail.append(self.pos.copy())
        direction = target_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        self.pos += direction * self.speed

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human_playable"]}

    game_description = (
        "Defend your deep-sea energy tower from waves of menacing anglerfish. "
        "Use powerful sonar pings to create chain reactions and survive as long as you can."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ to move the tower. Press Shift to flip your active cannon "
        "between the top and bottom. Press Space to fire a sonar pulse."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_WAVE = 50

    # Colors
    COLOR_BG_TOP = (1, 5, 25)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_PLAYER_ACTIVE = (100, 255, 255)
    COLOR_ANGLER_LURE = (255, 50, 0)
    COLOR_ANGLER_BODY = (150, 30, 0)
    COLOR_SONAR_PRIMARY = (0, 255, 150)
    COLOR_SONAR_CHAIN = (200, 255, 0)
    COLOR_UI_TEXT = (200, 200, 255)
    COLOR_ENERGY_BAR = (0, 200, 255)
    COLOR_ENERGY_BG = (50, 50, 80)

    # Player
    PLAYER_TOWER_WIDTH = 12
    PLAYER_TOWER_HEIGHT = 100
    PLAYER_SPEED = 5.0

    # Energy
    ENERGY_MAX = 100
    ENERGY_REGEN = 0.2
    SONAR_COST = 40

    # Waves
    WAVE_DURATION = 15 * FPS # 15 seconds
    INTER_WAVE_DURATION = 5 * FPS # 5 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.render_mode = render_mode

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This is for dev; remove from final

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_y = self.HEIGHT / 2
        self.player_active_end = 1  # 1 for bottom, -1 for top

        self.energy = self.ENERGY_MAX
        self.last_space_press = False
        self.last_shift_press = False

        self.wave = 1
        self.wave_phase_timer = self.INTER_WAVE_DURATION
        self.is_inter_wave = True

        self.anglerfish = []
        self.sonar_pulses = []
        self.particles = []

        self._update_wave_difficulty()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle Input & Player State ---
        self._handle_input(action)
        self.energy = min(self.ENERGY_MAX, self.energy + self.ENERGY_REGEN)

        # --- Update Game Logic ---
        self._update_waves()
        self._update_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check Termination Conditions ---
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if self.wave > self.WIN_WAVE:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100

        if self.game_over and not self.win:
            terminated = True
            reward = -100

        if terminated and self.is_inter_wave and self.wave > 1:
             reward += 1.0 * (self.wave -1)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement (up/down)
        if movement == 1:  # Up
            self.player_y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_y += self.PLAYER_SPEED
        self.player_y = np.clip(self.player_y, self.PLAYER_TOWER_HEIGHT / 2, self.HEIGHT - self.PLAYER_TOWER_HEIGHT / 2)

        # Gravity Flip (Shift)
        if shift_held and not self.last_shift_press:
            self.player_active_end *= -1
            self._create_particles(self._get_player_tower_center(), 20, self.COLOR_PLAYER_GLOW, (0, 3), (10, 20))
        self.last_shift_press = shift_held

        # Sonar (Space)
        if space_held and not self.last_space_press and self.energy >= self.SONAR_COST:
            self.energy -= self.SONAR_COST
            active_end_pos = self._get_player_active_end_pos()
            self.sonar_pulses.append(SonarPulse(active_end_pos, 200, 4, self.COLOR_SONAR_PRIMARY, 60, 0))
        self.last_space_press = space_held
        
    def _update_waves(self):
        self.wave_phase_timer -= 1
        if self.wave_phase_timer <= 0:
            self.is_inter_wave = not self.is_inter_wave
            if self.is_inter_wave:
                self.wave_phase_timer = self.INTER_WAVE_DURATION
                if not self.game_over:
                    self.wave += 1
                    self._update_wave_difficulty()
            else: # Wave starts
                self.wave_phase_timer = self.WAVE_DURATION
        
        if not self.is_inter_wave:
            self.angler_spawn_timer -= 1
            if self.angler_spawn_timer <= 0:
                self._spawn_anglerfish()
                self.angler_spawn_timer = self.current_angler_spawn_rate

    def _update_wave_difficulty(self):
        difficulty_mod = (1.05 ** math.floor((self.wave - 1) / 5))
        self.current_angler_speed = 1.0 * difficulty_mod
        self.current_angler_spawn_rate = max(10, 30 / difficulty_mod)
        self.angler_spawn_timer = 0

    def _spawn_anglerfish(self):
        side = random.choice(['left', 'right', 'top', 'bottom'])
        if side == 'left':
            pos = [-20, random.uniform(0, self.HEIGHT)]
        elif side == 'right':
            pos = [self.WIDTH + 20, random.uniform(0, self.HEIGHT)]
        elif side == 'top':
            pos = [random.uniform(0, self.WIDTH), -20]
        else: # bottom
            pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 20]
        
        fish = Anglerfish(pos, self.current_angler_speed, self.COLOR_ANGLER_LURE, self.COLOR_ANGLER_BODY)
        self.anglerfish.append(fish)

    def _update_entities(self):
        target_pos = self._get_player_active_end_pos()
        for fish in self.anglerfish:
            fish.update(target_pos)

        self.sonar_pulses = [p for p in self.sonar_pulses if p.update()]
        self.particles = [p for p in self.particles if p.update()]

    def _handle_collisions(self):
        reward = 0
        newly_hit_fish_indices = set()

        for pulse in self.sonar_pulses:
            for i, fish in enumerate(self.anglerfish):
                if i in newly_hit_fish_indices:
                    continue
                dist = np.linalg.norm(fish.pos - pulse.origin)
                if dist < pulse.radius and i not in pulse.hit_fish:
                    newly_hit_fish_indices.add(i)
                    pulse.hit_fish.add(i)
        
        if newly_hit_fish_indices:
            for i in sorted(list(newly_hit_fish_indices), reverse=True):
                fish = self.anglerfish.pop(i)
                reward += 0.1
                self.score += 10
                self._create_particles(fish.pos, 30, self.COLOR_ANGLER_LURE, (0, 4), (15, 30))
                # Chain reaction
                self.sonar_pulses.append(SonarPulse(fish.pos, 75, 6, self.COLOR_SONAR_CHAIN, 25, 1))

        # Anglerfish vs Player Tower
        tower_rect = self._get_player_tower_rect()
        for fish in self.anglerfish:
            if tower_rect.collidepoint(fish.pos):
                self.game_over = True
                self._create_particles(fish.pos, 100, self.COLOR_PLAYER_ACTIVE, (0, 6), (30, 60))
                break
        
        return reward

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            s = pygame.Surface((p.radius * 2, p.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.radius, p.radius), p.radius)
            self.screen.blit(s, (int(p.pos[0] - p.radius), int(p.pos[1] - p.radius)))

        for pulse in self.sonar_pulses:
            alpha = pulse.get_alpha()
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pulse.origin[0]), int(pulse.origin[1]), int(pulse.radius), (*pulse.color, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(pulse.origin[0]), int(pulse.origin[1]), int(pulse.radius)-1, (*pulse.color, alpha))

        for fish in self.anglerfish:
            # Trail
            for i, p in enumerate(fish.trail):
                alpha = int(80 * (i / len(fish.trail)))
                pygame.draw.circle(self.screen, (*fish.body_color, alpha), (int(p[0]), int(p[1])), 2, 1)
            # Body
            pygame.draw.circle(self.screen, fish.body_color, (int(fish.pos[0]), int(fish.pos[1])), 4)
            # Lure
            pygame.draw.circle(self.screen, fish.lure_color, (int(fish.pos[0]), int(fish.pos[1])), 6)
            pygame.gfxdraw.aacircle(self.screen, int(fish.pos[0]), int(fish.pos[1]), 6, fish.lure_color)

        self._render_player()

    def _render_player(self):
        tower_rect = self._get_player_tower_rect()
        active_end_pos = self._get_player_active_end_pos()

        # Glow effect
        for i in range(5, 0, -1):
            glow_rect = tower_rect.inflate(i * 4, i * 4)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_PLAYER_GLOW, 15))
            self.screen.blit(s, glow_rect.topleft)

        # Main tower
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, tower_rect, border_radius=4)
        
        # Active end
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_ACTIVE, (int(active_end_pos[0]), int(active_end_pos[1])), 8)
        pygame.gfxdraw.aacircle(self.screen, int(active_end_pos[0]), int(active_end_pos[1]), 8, self.COLOR_PLAYER_ACTIVE)

    def _render_ui(self):
        # Wave and Score
        wave_text = self.font_ui.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(score_text, (10, 35))

        # Energy Bar
        bar_width, bar_height = 150, 20
        bar_x, bar_y = self.WIDTH - bar_width - 10, 10
        energy_ratio = self.energy / self.ENERGY_MAX
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, bar_width * energy_ratio, bar_height), border_radius=4)

        # Wave Status
        if self.is_inter_wave and self.wave <= self.WIN_WAVE:
            timer_secs = self.wave_phase_timer / self.FPS
            status_text = self.font_title.render(f"WAVE {self.wave} IN {timer_secs:.1f}S", True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_text, text_rect)

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER_ACTIVE
            else:
                msg = "TOWER BREACHED"
                color = self.COLOR_ANGLER_LURE
            
            end_text = self.font_title.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_player_tower_rect(self):
        return pygame.Rect(
            self.WIDTH / 2 - self.PLAYER_TOWER_WIDTH / 2,
            self.player_y - self.PLAYER_TOWER_HEIGHT / 2,
            self.PLAYER_TOWER_WIDTH,
            self.PLAYER_TOWER_HEIGHT
        )
    
    def _get_player_tower_center(self):
        return np.array([self.WIDTH/2, self.player_y])

    def _get_player_active_end_pos(self):
        return np.array([self.WIDTH / 2, self.player_y + self.player_active_end * self.PLAYER_TOWER_HEIGHT / 2])

    def _create_particles(self, origin, num, color, speed_range, life_range):
        for _ in range(num):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(*life_range)
            radius = random.uniform(1, 4)
            self.particles.append(Particle(origin, vel, radius, color, life))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    # This is conditional on the render_mode being 'human_playable'
    if env.render_mode == "human_playable":
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Submarine Tower Defense")
        clock = pygame.time.Clock()

    while not done:
        action = [0, 0, 0] # Default no-op action
        # --- Action mapping for human player ---
        if env.render_mode == "human_playable":
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                movement = 1 # up
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                movement = 2 # down
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame rendering ---
        if env.render_mode == "human_playable":
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(GameEnv.FPS)
    
    print(f"Game Over! Final Info: {info}")
    env.close()