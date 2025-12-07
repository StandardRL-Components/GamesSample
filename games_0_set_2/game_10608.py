import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:39:16.911520
# Source Brief: brief_00608.md
# Brief Index: 608
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Ocean's Heart'.

    In this idle clicker game, the agent must harness the energy of the Ocean's
    Heart by strategically placing and upgrading portals. The goal is to collect
    10,000 energy units before the episode ends.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): Selects one of four portal locations.
        - 0: No-op
        - 1: Select UP portal
        - 2: Select DOWN portal
        - 3: Select LEFT portal
        - 4: Select RIGHT portal
    - `actions[1]` (Space): Taps to strengthen the magnetic field, providing a
      temporary boost to energy generation.
        - 0: Released
        - 1: Held/Pressed
    - `actions[2]` (Shift): Upgrades the currently selected portal if affordable.
        - 0: Released
        - 1: Held/Pressed

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +0.1 for every 10 energy collected.
    - +1.0 for each portal upgrade.
    - +5.0 for unlocking a new portal region.
    - +100.0 for winning the game (reaching 10,000 energy).

    **Termination:**
    - Reaching 10,000 total energy.
    - Reaching the maximum step limit of 5000.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Harness the energy of the Ocean's Heart by placing and upgrading portals to collect energy in this idle clicker game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a portal. Press space to boost energy generation and hold shift to upgrade the selected portal."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_SCORE = 10000
        self.MAX_STEPS = 5000

        # --- Gameplay Parameters ---
        self.BASE_ENERGY_PER_STEP = 0.01
        self.PORTAL_BASE_ENERGY = 0.05
        self.PORTAL_UPGRADE_COST_BASE = 20
        self.PORTAL_UPGRADE_COST_MULT = 1.5
        self.TAP_BOOST_DURATION = 15  # steps
        self.TAP_BOOST_MULTIPLIER = 3.0
        self.REGION_UNLOCK_THRESHOLDS = [100, 500, 2000]

        # --- Visuals ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_WAVE = (20, 35, 60)
        self.COLOR_HEART = (255, 200, 80)
        self.COLOR_PORTAL = (180, 100, 255)
        self.COLOR_ENERGY = (255, 220, 120)
        self.COLOR_AURA = (220, 220, 255)
        self.COLOR_TAP = (255, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_ACCENT = (255, 200, 80)
        self.COLOR_SELECTOR = (100, 255, 100)

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
        try:
            self.font_ui = pygame.font.SysFont("consolas", 20)
            self.font_portal = pygame.font.SysFont("consolas", 14)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_portal = pygame.font.SysFont(None, 18)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.portals = None
        self.selected_portal_idx = None
        self.energy_particles = None
        self.tap_effects = None
        self.tap_boost_timer = None
        self.previous_score_reward_check = None
        self.center_pos = None

        # self.reset() # This is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.previous_score_reward_check = 0

        self.center_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        portal_offsets = [
            (0, -120), (0, 120), (-180, 0), (180, 0) # Up, Down, Left, Right
        ]
        self.portals = [
            {
                "pos": (self.center_pos[0] + off[0], self.center_pos[1] + off[1]),
                "level": 0,
                "unlocked": (i == 0), # Start with one region unlocked
                "aura_phase": self.np_random.uniform(0, 2 * math.pi)
            } for i, off in enumerate(portal_offsets)
        ]

        self.selected_portal_idx = 0
        self.energy_particles = []
        self.tap_effects = []
        self.tap_boost_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle Actions
        reward += self._handle_actions(action)

        # 2. Update Game State
        self._update_game_state()

        # 3. Check for Milestones
        reward += self._check_milestones()

        # 4. Calculate passive energy reward
        reward_units_earned = int(self.score // 10) - int(self.previous_score_reward_check // 10)
        if reward_units_earned > 0:
            reward += reward_units_earned * 0.1
            self.previous_score_reward_check = self.score

        # 5. Check for Termination
        terminated = (self.score >= self.WIN_SCORE)
        truncated = (self.steps >= self.MAX_STEPS)
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Victory bonus

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = 0

        # Movement: Select Portal
        if 1 <= movement <= 4:
            self.selected_portal_idx = movement - 1

        # Space: Tap for Boost
        if space_held:
            # // SFX: Tap sound
            self.tap_boost_timer = self.TAP_BOOST_DURATION
            self.tap_effects.append({
                "pos": self.center_pos, "radius": 10, "alpha": 255, "life": 20
            })

        # Shift: Upgrade Portal
        if shift_held:
            portal = self.portals[self.selected_portal_idx]
            if portal["unlocked"]:
                cost = self._calculate_upgrade_cost(portal["level"])
                if self.score >= cost:
                    # // SFX: Upgrade sound
                    self.score -= cost
                    portal["level"] += 1
                    action_reward += 1.0

        return action_reward

    def _update_game_state(self):
        # Calculate energy per step
        energy_this_step = self.BASE_ENERGY_PER_STEP
        for portal in self.portals:
            if portal["level"] > 0:
                energy_this_step += portal["level"] * self.PORTAL_BASE_ENERGY

        if self.tap_boost_timer > 0:
            energy_this_step *= self.TAP_BOOST_MULTIPLIER
            self.tap_boost_timer -= 1

        self.score += energy_this_step

        # Update particles
        if self.steps % 2 == 0:
            for i, portal in enumerate(self.portals):
                if portal["level"] > 0:
                    num_particles = min(5, portal['level'])
                    for _ in range(num_particles):
                        self._spawn_particle(i)

        for p in self.energy_particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.energy_particles = [p for p in self.energy_particles if p['life'] > 0]

        # Update tap effects
        for e in self.tap_effects:
            e['radius'] += 5
            e['alpha'] = max(0, e['alpha'] - 12)
            e['life'] -= 1
        self.tap_effects = [e for e in self.tap_effects if e['life'] > 0]

        # Update portal auras
        for portal in self.portals:
            portal["aura_phase"] += 0.05

    def _check_milestones(self):
        milestone_reward = 0
        # Check for region unlocks
        if not self.portals[1]["unlocked"] and self.score >= self.REGION_UNLOCK_THRESHOLDS[0]:
            self.portals[1]["unlocked"] = True
            milestone_reward += 5.0 # // SFX: Unlock sound
        if not self.portals[2]["unlocked"] and self.score >= self.REGION_UNLOCK_THRESHOLDS[1]:
            self.portals[2]["unlocked"] = True
            milestone_reward += 5.0
        if not self.portals[3]["unlocked"] and self.score >= self.REGION_UNLOCK_THRESHOLDS[2]:
            self.portals[3]["unlocked"] = True
            milestone_reward += 5.0
        return milestone_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Animated waves
        for i in range(15):
            y_offset = i * 30
            points = []
            for x in range(0, self.WIDTH + 1, 10):
                y = y_offset + math.sin(self.steps * 0.02 + x * 0.02) * 5
                points.append((x, int(y)))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_WAVE, False, points)

    def _render_game_elements(self):
        # Render particles first (background layer)
        for p in self.energy_particles:
            alpha = max(0, min(255, int(p['life'] * 10)))
            color = (*self.COLOR_ENERGY, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['size'], color)

        # Render Ocean's Heart
        heart_pulse = 5 + math.sin(self.steps * 0.05) * 3
        self._draw_glow_circle(self.screen, self.COLOR_HEART, self.center_pos, 25 + heart_pulse, 30)

        # Render Portals and Auras
        for i, portal in enumerate(self.portals):
            if portal["level"] > 0 or portal["unlocked"]:
                base_radius = 10 + portal["level"] * 2
                self._draw_glow_circle(self.screen, self.COLOR_PORTAL, portal["pos"], base_radius, 20)

                if portal["level"] > 0:
                    aura_radius = base_radius + 15 + math.sin(portal["aura_phase"]) * 5
                    self._draw_glow_circle(self.screen, self.COLOR_AURA, portal["pos"], aura_radius, 15, hollow=True)

        # Render Tap Effects
        for e in self.tap_effects:
            if e['alpha'] > 0:
                self._draw_glow_circle(self.screen, (*self.COLOR_TAP, int(e['alpha'])), e['pos'], e['radius'], 10, hollow=True)

        # Render Selector
        selected_portal = self.portals[self.selected_portal_idx]
        if selected_portal["unlocked"]:
            angle = (self.steps * 0.1) % (2 * math.pi)
            sel_radius = 20 + selected_portal["level"] * 2 + 15
            for i in range(3):
                a = angle + i * 2 * math.pi / 3
                p1 = (int(selected_portal["pos"][0] + sel_radius * math.cos(a)),
                      int(selected_portal["pos"][1] + sel_radius * math.sin(a)))
                p2 = (int(selected_portal["pos"][0] + (sel_radius + 10) * math.cos(a)),
                      int(selected_portal["pos"][1] + (sel_radius + 10) * math.sin(a)))
                pygame.draw.line(self.screen, self.COLOR_SELECTOR, p1, p2, 2)


    def _render_ui(self):
        # Total Energy (Score)
        score_text = f"ENERGY: {int(self.score):,}"
        self._draw_text(score_text, self.font_ui, self.COLOR_UI_TEXT, self.screen, (10, 10))

        # Energy Rate
        rate = self.BASE_ENERGY_PER_STEP
        for p in self.portals:
            rate += p["level"] * self.PORTAL_BASE_ENERGY
        if self.tap_boost_timer > 0:
            rate *= self.TAP_BOOST_MULTIPLIER
        rate_text = f"RATE: {rate * self.FPS:.2f}/s"
        if self.tap_boost_timer > 0:
            rate_text += " (BOOST!)"
        text_surface = self.font_ui.render(rate_text, True, self.COLOR_UI_ACCENT if self.tap_boost_timer > 0 else self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        # Portal Info
        for portal in self.portals:
            if portal["unlocked"]:
                level_text = f"LVL: {portal['level']}"
                self._draw_text(level_text, self.font_portal, self.COLOR_UI_TEXT, self.screen, (portal["pos"][0], portal["pos"][1] - 40), center=True)
                cost = self._calculate_upgrade_cost(portal['level'])
                cost_text = f"UPG: {int(cost)}"
                cost_color = self.COLOR_UI_ACCENT if self.score >= cost else (150, 150, 150)
                self._draw_text(cost_text, self.font_portal, cost_color, self.screen, (portal["pos"][0], portal["pos"][1] - 25), center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "portals_level": [p['level'] for p in self.portals],
            "tap_boost_active": self.tap_boost_timer > 0
        }

    # --- Helper Functions ---

    def _calculate_upgrade_cost(self, level):
        return self.PORTAL_UPGRADE_COST_BASE * (self.PORTAL_UPGRADE_COST_MULT ** level)

    def _spawn_particle(self, portal_idx):
        if len(self.energy_particles) > 300: return
        portal_pos = self.portals[portal_idx]["pos"]
        start_pos = [self.center_pos[0] + self.np_random.uniform(-5, 5), self.center_pos[1] + self.np_random.uniform(-5, 5)]
        direction = (portal_pos[0] - start_pos[0], portal_pos[1] - start_pos[1])
        dist = math.hypot(*direction)
        if dist == 0: return
        vel = [d / dist * 3 for d in direction]
        self.energy_particles.append({
            "pos": start_pos,
            "vel": vel,
            "life": int(dist / 3) + 10,
            "size": self.np_random.integers(1, 4)
        })

    def _draw_text(self, text, font, color, surface, pos, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        surface.blit(text_surface, text_rect)

    def _draw_glow_circle(self, surface, color, center, radius, glow_width, hollow=False):
        center_int = (int(center[0]), int(center[1]))
        radius = max(1, int(radius))
        glow_width = max(1, int(glow_width))

        for i in range(glow_width, 0, -1):
            alpha = int(color[3] if len(color) == 4 else 255)
            alpha = int(alpha * (1 - (i / glow_width))**1.5)
            glow_color = (*color[:3], alpha)
            current_radius = radius + i
            pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], current_radius, glow_color)
        
        if not hollow:
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    
    # Create a display surface if we are playing directly
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ocean's Heart")

    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("----------------\n")

    while running:
        # --- Human Input to Action Mapping ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Render to Display ---
        # The observation is (H, W, C), but pygame blit needs a surface
        # We can just re-get the internal surface from the env
        surf = pygame.transform.rotate(env.screen, -90)
        surf = pygame.transform.flip(surf, True, False)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()