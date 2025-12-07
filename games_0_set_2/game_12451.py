import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Higgs Boson Hunter Gymnasium Environment.

    The player controls a targeting reticle to find and capture Higgs bosons
    in a dynamic energy field. Capturing bosons provides points that can be
    used to upgrade abilities. The goal is to capture 10 bosons before
    running out of energy.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) - Fire energy pulse
    - actions[2]: Shift button (0=released, 1=held) - Spend skill points on upgrades

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +5 for capturing a Higgs boson.
    - +0.1 per step if energy > 50%.
    - -0.1 per step if energy < 20%.
    - +100 for winning (capturing 10 bosons).
    - -100 for losing (energy depleted).
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a targeting reticle to find and capture elusive Higgs bosons in a dynamic energy field. "
        "Use captured points to upgrade your abilities and capture 10 bosons to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to fire an energy pulse and capture bosons. "
        "Press shift to spend skill points on upgrades."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    WIN_CONDITION = 10

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PARTICLE = (30, 40, 70)
    COLOR_RETICLE = (0, 255, 255)
    COLOR_PULSE = (255, 255, 0)
    COLOR_BOSON = (255, 220, 100)
    COLOR_BOSON_GLOW = (255, 180, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_HIGH = (40, 200, 40)
    COLOR_ENERGY_LOW = (255, 40, 40)
    COLOR_ENERGY_MID = (255, 200, 40)
    COLOR_UI_BG = (50, 60, 80, 150)

    # Game Parameters
    RETICLE_SPEED = 10
    INITIAL_ENERGY = 100.0
    
    BASE_PULSE_COST = 15.0
    BASE_PULSE_RADIUS = 60
    BASE_ENERGY_REGEN = 0.1

    BOSON_SPAWN_INTERVAL = 5 * FPS # 5 seconds
    BOSON_INITIAL_LIFETIME = 3 * FPS # 3 seconds

    # Upgrade Parameters
    UPGRADE_COST = 3
    COST_REDUCTION_PER_LEVEL = 2.0
    RADIUS_INCREASE_PER_LEVEL = 15
    REGEN_INCREASE_PER_LEVEL = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reticle_pos = None
        self.energy = 0.0
        self.bosons_captured = 0
        self.skill_points = 0
        self.skill_levels = {}
        self.higgs_bosons = []
        self.energy_pulses = []
        self.background_particles = []
        self.boson_spawn_timer = 0
        self.boson_visibility_duration = 0
        self.last_space_press = False
        self.last_shift_press = False
        self.next_upgrade_type = 0
        self.last_upgrade_msg = ""
        self.last_upgrade_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reticle_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.energy = self.INITIAL_ENERGY
        self.bosons_captured = 0
        self.skill_points = 0
        
        self.skill_levels = {"cost": 0, "radius": 0, "regen": 0}
        self.next_upgrade_type = 0 # 0: cost, 1: radius, 2: regen

        self.higgs_bosons = []
        self.energy_pulses = []
        self.background_particles = [self._create_particle() for _ in range(150)]
        
        self.boson_spawn_timer = self.BOSON_SPAWN_INTERVAL
        self.boson_visibility_duration = self.BOSON_INITIAL_LIFETIME

        self.last_space_press = False
        self.last_shift_press = False
        self.last_upgrade_msg = ""
        self.last_upgrade_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_game_state()
        
        # --- Calculate Reward ---
        # Event-based rewards are handled in _update_game_state
        if self.energy > 50:
            reward += 0.1
        elif self.energy < 20:
            reward -= 0.1

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.bosons_captured >= self.WIN_CONDITION:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.energy <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED
        
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.HEIGHT)

        # --- Fire Pulse (Space) ---
        if space_held and not self.last_space_press:
            pulse_cost = self._get_pulse_cost()
            if self.energy >= pulse_cost:
                self.energy -= pulse_cost
                self.energy_pulses.append({
                    'pos': pygame.math.Vector2(self.reticle_pos),
                    'radius': 0,
                    'max_radius': self._get_pulse_radius(),
                    'lifetime': 0.6 * self.FPS, # 0.6 seconds
                    'max_lifetime': 0.6 * self.FPS
                })
        self.last_space_press = space_held

        # --- Upgrade (Shift) ---
        if shift_held and not self.last_shift_press:
            if self.skill_points >= self.UPGRADE_COST:
                self.skill_points -= self.UPGRADE_COST
                self._apply_upgrade()
        self.last_shift_press = shift_held

    def _update_game_state(self):
        # --- Energy Regeneration ---
        self.energy = min(self.INITIAL_ENERGY, self.energy + self._get_energy_regen())

        # --- Update Pulses ---
        for pulse in self.energy_pulses:
            pulse['lifetime'] -= 1
            pulse['radius'] = (1 - (pulse['lifetime'] / pulse['max_lifetime'])) * pulse['max_radius']
        self.energy_pulses = [p for p in self.energy_pulses if p['lifetime'] > 0]

        # --- Update Bosons ---
        for boson in self.higgs_bosons:
            boson['lifetime'] -= 1
        
        # --- Collision Detection ---
        captured_bosons = []
        for boson in self.higgs_bosons:
            for pulse in self.energy_pulses:
                if boson['pos'].distance_to(pulse['pos']) < pulse['radius']:
                    captured_bosons.append(boson)
                    break
        
        if captured_bosons:
            for boson in captured_bosons:
                if boson in self.higgs_bosons: # Ensure not already removed
                    self.higgs_bosons.remove(boson)
                    self.bosons_captured += 1
                    self.score += 5 # Event-based reward
                    self.skill_points += 1
                    
                    # Difficulty scaling
                    if self.bosons_captured > 0 and self.bosons_captured % 2 == 0:
                        self.boson_visibility_duration = max(1 * self.FPS, self.boson_visibility_duration - 0.1 * self.FPS)

        # Remove expired bosons
        self.higgs_bosons = [b for b in self.higgs_bosons if b['lifetime'] > 0]

        # --- Boson Spawning ---
        self.boson_spawn_timer -= 1
        if self.boson_spawn_timer <= 0:
            self.boson_spawn_timer = self.BOSON_SPAWN_INTERVAL
            self.higgs_bosons.append({
                'pos': pygame.math.Vector2(
                    random.uniform(50, self.WIDTH - 50),
                    random.uniform(50, self.HEIGHT - 50)
                ),
                'lifetime': self.boson_visibility_duration,
                'max_lifetime': self.boson_visibility_duration,
                'radius': 12,
                'phase': random.uniform(0, 2 * math.pi) # For pulsation
            })

        # --- Update Background ---
        for p in self.background_particles:
            p['pos'] += p['vel']
            if p['pos'].x < 0 or p['pos'].x > self.WIDTH or p['pos'].y < 0 or p['pos'].y > self.HEIGHT:
                p.update(self._create_particle(on_edge=True))

        # --- Update UI timers ---
        if self.last_upgrade_timer > 0:
            self.last_upgrade_timer -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Background Particles ---
        for p in self.background_particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), 
                (*self.COLOR_PARTICLE, p['alpha'])
            )

        # --- Higgs Bosons ---
        for boson in self.higgs_bosons:
            t = 1 - (boson['lifetime'] / boson['max_lifetime'])
            pulse_factor = 1.0 + 0.2 * math.sin(boson['phase'] + self.steps * 0.2)
            current_radius = int(boson['radius'] * pulse_factor)
            
            # Fade in/out effect
            alpha_fade = min(1.0, t * 5.0) * min(1.0, (boson['lifetime'] / (0.5 * self.FPS)))
            
            # Glow
            glow_radius = int(current_radius * 2.5)
            glow_alpha = int(90 * alpha_fade * pulse_factor)
            if glow_alpha > 0:
                self._draw_circle_alpha(self.COLOR_BOSON_GLOW, boson['pos'], glow_radius, glow_alpha)

            # Core
            core_alpha = int(255 * alpha_fade)
            if core_alpha > 0:
                self._draw_circle_alpha(self.COLOR_BOSON, boson['pos'], current_radius, core_alpha)

        # --- Energy Pulses ---
        for pulse in self.energy_pulses:
            alpha = int(255 * (pulse['lifetime'] / pulse['max_lifetime']))
            if alpha > 0:
                pygame.gfxdraw.aacircle(
                    self.screen, int(pulse['pos'].x), int(pulse['pos'].y), int(pulse['radius']),
                    (*self.COLOR_PULSE, alpha)
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(pulse['pos'].x), int(pulse['pos'].y), int(pulse['radius']-1),
                    (*self.COLOR_PULSE, alpha)
                )

        # --- Reticle ---
        x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 12, y), (x + 12, y), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - 12), (x, y + 12), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_RETICLE)

    def _render_ui(self):
        # --- Energy Bar ---
        bar_width = 200
        bar_height = 20
        energy_ratio = np.clip(self.energy / self.INITIAL_ENERGY, 0, 1)
        
        if energy_ratio < 0.2: color = self.COLOR_ENERGY_LOW
        elif energy_ratio < 0.5: color = self.COLOR_ENERGY_MID
        else: color = self.COLOR_ENERGY_HIGH
        
        pygame.draw.rect(self.screen, (30,30,30), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (10, 10, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)
        
        energy_text = self.font_main.render(f"ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (15, 35))

        # --- Bosons Captured ---
        boson_text = self.font_main.render(f"BOSONS: {self.bosons_captured} / {self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        self.screen.blit(boson_text, (self.WIDTH - boson_text.get_width() - 15, 15))

        # --- Upgrade Info ---
        upgrades_available = self.skill_points // self.UPGRADE_COST
        upgrade_text = f"UPGRADES READY: {upgrades_available}"
        color = self.COLOR_BOSON if upgrades_available > 0 else self.COLOR_UI_TEXT
        
        upgrade_surf = self.font_main.render(upgrade_text, True, color)
        self.screen.blit(upgrade_surf, ((self.WIDTH - upgrade_surf.get_width()) // 2, self.HEIGHT - 35))
        
        # Display last upgrade message
        if self.last_upgrade_timer > 0:
            alpha = min(255, self.last_upgrade_timer * 10)
            upgrade_msg_surf = self.font_small.render(self.last_upgrade_msg, True, self.COLOR_PULSE)
            upgrade_msg_surf.set_alpha(alpha)
            self.screen.blit(upgrade_msg_surf, ((self.WIDTH - upgrade_msg_surf.get_width()) // 2, self.HEIGHT - 60))

        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            if self.bosons_captured >= self.WIN_CONDITION:
                msg = "VICTORY"
                color = self.COLOR_BOSON
            elif self.energy <= 0:
                msg = "ENERGY DEPLETED"
                color = self.COLOR_ENERGY_LOW
            else:
                msg = "TIME LIMIT REACHED"
                color = self.COLOR_UI_TEXT

            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, ((self.WIDTH - end_text.get_width()) / 2, (self.HEIGHT - end_text.get_height()) / 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "bosons_captured": self.bosons_captured, "energy": self.energy}

    # --- Helper Methods ---
    def _create_particle(self, on_edge=False):
        if on_edge:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), -5)
            elif edge == 'bottom': pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + 5)
            elif edge == 'left': pos = pygame.math.Vector2(-5, random.uniform(0, self.HEIGHT))
            else: pos = pygame.math.Vector2(self.WIDTH + 5, random.uniform(0, self.HEIGHT))
        else:
            pos = pygame.math.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT))
        
        return {
            'pos': pos,
            'vel': pygame.math.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
            'size': random.uniform(1, 3),
            'alpha': random.randint(30, 80)
        }

    def _draw_circle_alpha(self, color, pos, radius, alpha):
        if radius <= 0: return
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
        self.screen.blit(s, (int(pos.x - radius), int(pos.y - radius)))

    def _get_pulse_cost(self):
        return max(5.0, self.BASE_PULSE_COST - self.skill_levels['cost'] * self.COST_REDUCTION_PER_LEVEL)

    def _get_pulse_radius(self):
        return self.BASE_PULSE_RADIUS + self.skill_levels['radius'] * self.RADIUS_INCREASE_PER_LEVEL

    def _get_energy_regen(self):
        return self.BASE_ENERGY_REGEN + self.skill_levels['regen'] * self.REGEN_INCREASE_PER_LEVEL

    def _apply_upgrade(self):
        upgrade_map = {0: "cost", 1: "radius", 2: "regen"}
        upgrade_name_map = {0: "Pulse Cost", 1: "Pulse Radius", 2: "Energy Regen"}
        
        upgrade_key = upgrade_map[self.next_upgrade_type]
        self.skill_levels[upgrade_key] += 1
        
        self.last_upgrade_msg = f"Upgraded: {upgrade_name_map[self.next_upgrade_type]} LVL {self.skill_levels[upgrade_key]}"
        self.last_upgrade_timer = 2 * self.FPS
        
        self.next_upgrade_type = (self.next_upgrade_type + 1) % 3

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame window for human play
    # Re-enable display for human play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Higgs Boson Hunter")
    clock = pygame.time.Clock()

    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the game
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    pygame.quit()