import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:02:07.115161
# Source Brief: brief_03365.md
# Brief Index: 3365
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment for "Atomic Nucleus Defense".

    This tower defense game challenges the agent to defend a central nucleus
    from incoming waves of mesons. The agent can place defensive lepton towers
    on concentric rings and manipulate the flow of time.

    **Visuals:**
    - Stylized, high-contrast particle physics aesthetic.
    - Glowing nucleus, towers, and effects.
    - Particle trails and explosion effects.
    - Smooth, interpolated motion.

    **Gameplay:**
    - Place towers on designated slots to shoot mesons.
    - Manage energy, which regenerates over time.
    - Use a time dilation ability to slow down enemies for strategic placement.
    - Survive increasing waves of enemies with scaling difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend the central nucleus from waves of incoming mesons by placing defensive towers and manipulating time."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a tower slot. Press space to build a tower and shift to toggle time dilation."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    FPS = 30
    MAX_STEPS = 2500
    TOTAL_WAVES = 15

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_NUCLEUS = (255, 220, 50)
    COLOR_TOWER = (0, 180, 255)
    COLOR_MESON = (255, 50, 100)
    COLOR_ENERGY = (50, 255, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_DILATION_OVERLAY = (0, 100, 150, 60) # RGBA

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 28, bold=True)

        # Game config
        self._config_game_parameters()

        # Initialize state variables
        # self.reset() is called by the environment wrapper

    def _config_game_parameters(self):
        """Set up gameplay constants."""
        self.initial_nucleus_health = 100
        self.max_energy = 100
        self.energy_regen_rate = 0.2  # Energy per step

        self.placement_rings = [
            {"radius": 80, "slots": 6},
            {"radius": 140, "slots": 8},
            {"radius": 200, "slots": 10},
        ]

        self.tower_types = {
            "ELECTRON": {"cost": 25, "damage": 2.5, "range": 100, "fire_rate": 15, "unlock_wave": 1},
            "MUON": {"cost": 50, "damage": 8, "range": 160, "fire_rate": 30, "unlock_wave": 5},
            "TAU": {"cost": 80, "damage": 15, "range": 220, "fire_rate": 45, "unlock_wave": 10},
        }

        self.time_dilation_factor = 0.4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        # Player state
        self.nucleus_health = self.initial_nucleus_health
        self.energy = 50.0

        # Wave management
        self.wave_number = 0
        self.wave_in_progress = False
        self.mesons_in_wave = 0
        self.meson_spawn_timer = 0
        self.time_until_next_wave = self.FPS * 3 # 3 seconds

        # Entities
        self.mesons = []
        self.towers = []
        self.particles = []
        self.lasers = []

        # Controls state
        self.selector_ring_idx = 0
        self.selector_slot_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.time_dilation_active = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        self._handle_input(movement, space_pressed, shift_pressed)

        # --- 2. Update Game Logic ---
        time_factor = self.time_dilation_factor if self.time_dilation_active else 1.0

        # Energy regeneration
        self.energy = min(self.max_energy, self.energy + self.energy_regen_rate)

        # Wave management
        wave_completion_reward = self._update_wave_system()
        reward += wave_completion_reward

        # Update entities
        damage_to_nucleus = self._update_mesons(time_factor)
        mesons_destroyed_count = self._update_towers()
        self._update_particles()
        self._update_lasers()

        # --- 3. Calculate Reward ---
        reward += mesons_destroyed_count * 0.1  # Reward for destroying mesons
        reward -= damage_to_nucleus * 0.5      # Penalty for taking damage
        self.score += mesons_destroyed_count * 10 # Update score for UI

        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        if self.nucleus_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 20.0 # Penalty for losing
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        elif self.victory:
            self.game_over = True
            terminated = True
            reward += 50.0 # Bonus for winning

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        """Process agent actions."""
        # Movement: Select tower slot
        if movement in [1, 2]: # Up/Down cycles through rings
            prev_idx = self.selector_ring_idx
            if movement == 1: self.selector_ring_idx = (self.selector_ring_idx - 1 + len(self.placement_rings)) % len(self.placement_rings)
            if movement == 2: self.selector_ring_idx = (self.selector_ring_idx + 1) % len(self.placement_rings)
            if prev_idx != self.selector_ring_idx: self.selector_slot_idx = 0 # Reset slot on ring change
        elif movement in [3, 4]: # Left/Right cycles through slots on a ring
            num_slots = self.placement_rings[self.selector_ring_idx]["slots"]
            if movement == 3: self.selector_slot_idx = (self.selector_slot_idx - 1 + num_slots) % num_slots
            if movement == 4: self.selector_slot_idx = (self.selector_slot_idx + 1) % num_slots

        # Action: Place tower
        if space_pressed:
            self._attempt_place_tower()

        # Special: Toggle time dilation
        if shift_pressed:
            self.time_dilation_active = not self.time_dilation_active
            # sfx: time_warp_on / time_warp_off

    def _attempt_place_tower(self):
        """Place a tower at the selected slot if possible."""
        ring_info = self.placement_rings[self.selector_ring_idx]
        slot_angle = (2 * math.pi / ring_info["slots"]) * self.selector_slot_idx
        pos = (
            self.CENTER[0] + ring_info["radius"] * math.cos(slot_angle),
            self.CENTER[1] + ring_info["radius"] * math.sin(slot_angle)
        )

        # Check if a tower already exists
        for t in self.towers:
            if np.linalg.norm(np.array(t['pos']) - np.array(pos)) < 1:
                # TODO: Implement upgrading
                return

        # Determine best available tower type
        available_towers = [t_name for t_name, t_data in self.tower_types.items() if self.wave_number >= t_data['unlock_wave']]
        if not available_towers: return
        
        # Select the most expensive tower the player can afford
        tower_to_build = None
        for t_name in reversed(sorted(available_towers, key=lambda n: self.tower_types[n]['cost'])):
            if self.energy >= self.tower_types[t_name]['cost']:
                tower_to_build = t_name
                break
        
        if tower_to_build:
            stats = self.tower_types[tower_to_build]
            self.energy -= stats['cost']
            self.towers.append({
                "pos": pos, "type": tower_to_build, "level": 1,
                "range": stats["range"], "damage": stats["damage"],
                "fire_rate": stats["fire_rate"], "cooldown": 0
            })
            # sfx: place_tower

    def _update_wave_system(self):
        """Manages spawning mesons and advancing waves."""
        if self.wave_in_progress:
            if self.mesons_in_wave > 0:
                self.meson_spawn_timer -= 1
                if self.meson_spawn_timer <= 0:
                    self._spawn_meson()
                    self.mesons_in_wave -= 1
                    self.meson_spawn_timer = max(10, 40 - self.wave_number * 2)
            elif not self.mesons:
                self.wave_in_progress = False
                self.time_until_next_wave = self.FPS * 5 # 5 seconds
                if self.wave_number >= self.TOTAL_WAVES:
                    self.victory = True
                    return 0 # Final victory reward is handled in step()
                return 2.0 # Wave complete reward
        else:
            self.time_until_next_wave -= 1
            if self.time_until_next_wave <= 0:
                self._start_next_wave()
        return 0.0

    def _start_next_wave(self):
        """Initializes the next wave of enemies."""
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES: return

        self.wave_in_progress = True
        self.mesons_in_wave = 5 + self.wave_number * 2
        self.meson_spawn_timer = 0

    def _spawn_meson(self):
        """Creates a single meson at the edge of the screen."""
        edge = self.np_random.integers(4)
        if edge == 0: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -10
        elif edge == 1: x, y = self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif edge == 2: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10
        else: x, y = -10, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        pos = np.array([x, y], dtype=float)
        
        target_offset = self.np_random.uniform(-20, 20, 2)
        direction = (np.array(self.CENTER) + target_offset) - pos
        direction = direction / np.linalg.norm(direction)

        speed = 1.0 + self.wave_number * 0.2
        health = 10 + (self.wave_number // 5) * 10
        
        self.mesons.append({
            "pos": pos,
            "vel": direction * speed,
            "health": health,
            "max_health": health,
            "trail": [pos.copy() for _ in range(5)]
        })
        # sfx: meson_spawn

    def _update_mesons(self, time_factor):
        """Moves mesons and checks for collisions with the nucleus."""
        damage_dealt = 0
        for m in reversed(self.mesons):
            # Update trail
            m["trail"].pop(0)
            m["trail"].append(m["pos"].copy())
            
            # Move meson
            m["pos"] += m["vel"] * time_factor

            # Check for nucleus collision
            if np.linalg.norm(m["pos"] - self.CENTER) < 30:
                damage = m["health"] / 2 # Deal damage based on remaining health
                self.nucleus_health -= damage
                damage_dealt += damage
                self.mesons.remove(m)
                self._create_explosion(np.array(self.CENTER), self.COLOR_NUCLEUS, 30)
                # sfx: nucleus_hit
                continue

            if m["health"] <= 0:
                self.score += 5 * self.wave_number
                self._create_explosion(m["pos"], self.COLOR_MESON, 20)
                self.mesons.remove(m)
                # sfx: meson_destroy
        return damage_dealt

    def _update_towers(self):
        """Makes towers target and shoot at mesons."""
        mesons_destroyed_this_step = 0
        for t in self.towers:
            t["cooldown"] = max(0, t["cooldown"] - 1)
            if t["cooldown"] > 0:
                continue

            # Find target
            target = None
            min_dist = t["range"]
            for m in self.mesons:
                dist = np.linalg.norm(m["pos"] - np.array(t["pos"]))
                if dist < min_dist:
                    min_dist = dist
                    target = m
            
            # Fire at target
            if target:
                target["health"] -= t["damage"]
                t["cooldown"] = t["fire_rate"]
                self.lasers.append({"start": t["pos"], "end": target["pos"], "life": 3})
                # sfx: tower_shoot
                if target["health"] <= 0:
                    mesons_destroyed_this_step += 1
        return mesons_destroyed_this_step

    def _update_particles(self):
        """Animates and removes old particles."""
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_lasers(self):
        """Fades out laser effects."""
        for l in reversed(self.lasers):
            l["life"] -= 1
            if l["life"] <= 0:
                self.lasers.remove(l)

    def _create_explosion(self, pos, color, num_particles):
        """Spawns a particle explosion effect."""
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 25),
                "color": color
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Draws all the in-world game elements."""
        # Placement rings
        for ring in self.placement_rings:
            pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], ring["radius"], (40, 60, 90))

        # Lasers
        for l in self.lasers:
            alpha = max(0, (l['life'] / 3) * 255)
            color = self.COLOR_TOWER + (int(alpha),)
            pygame.draw.aaline(self.screen, color, l['start'], l['end'], 2)

        # Nucleus
        self._draw_glow_circle(self.screen, self.COLOR_NUCLEUS, self.CENTER, 25, 15)

        # Mesons and trails
        for m in self.mesons:
            # Trail
            for i, p in enumerate(m["trail"]):
                alpha = int(255 * (i / len(m["trail"]))**2 * 0.5)
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 3, self.COLOR_MESON + (alpha,))
            # Meson body
            self._draw_glow_circle(self.screen, self.COLOR_MESON, m["pos"], 5, 8)
            # Health bar
            if m['health'] < m['max_health']:
                bar_len = 15
                health_pct = m['health'] / m['max_health']
                pygame.draw.rect(self.screen, (50,50,50), (m['pos'][0] - bar_len/2, m['pos'][1] - 15, bar_len, 3))
                pygame.draw.rect(self.screen, self.COLOR_MESON, (m['pos'][0] - bar_len/2, m['pos'][1] - 15, bar_len * health_pct, 3))

        # Towers
        for t in self.towers:
            stats = self.tower_types[t["type"]]
            self._draw_glow_circle(self.screen, self.COLOR_TOWER, t["pos"], 8, 10)
            # Range indicator
            if self.time_dilation_active: # Show range only in slow-mo for clarity
                 pygame.gfxdraw.aacircle(self.screen, int(t['pos'][0]), int(t['pos'][1]), stats['range'], self.COLOR_TOWER + (50,))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 25))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, p["color"] + (alpha,))

        # Selector
        ring_info = self.placement_rings[self.selector_ring_idx]
        slot_angle = (2 * math.pi / ring_info["slots"]) * self.selector_slot_idx
        pos = (
            self.CENTER[0] + ring_info["radius"] * math.cos(slot_angle),
            self.CENTER[1] + ring_info["radius"] * math.sin(slot_angle)
        )
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        radius = int(12 + pulse * 4)
        alpha = int(150 + pulse * 105)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_SELECTOR + (alpha,))

        # Time Dilation Overlay
        if self.time_dilation_active:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_DILATION_OVERLAY)
            self.screen.blit(overlay, (0, 0))

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        """Renders a circle with a soft glow effect."""
        cen_x, cen_y = int(center[0]), int(center[1])
        for i in range(glow_strength, 0, -1):
            alpha = int(150 * (1 - (i / glow_strength))**2)
            pygame.gfxdraw.filled_circle(surface, cen_x, cen_y, radius + i, color + (alpha,))
        pygame.gfxdraw.filled_circle(surface, cen_x, cen_y, radius, color)
        pygame.gfxdraw.aacircle(surface, cen_x, cen_y, radius, color)

    def _render_ui(self):
        """Draws the UI overlay."""
        # Health
        health_text = self.font_large.render(f"NUCLEUS: {max(0, int(self.nucleus_health))}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Energy Bar
        bar_width, bar_height = 150, 20
        energy_pct = self.energy / self.max_energy
        pygame.draw.rect(self.screen, (40,40,60), (10, 40, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (10, 40, bar_width * energy_pct, bar_height))
        energy_text = self.font_small.render(f"ENERGY: {int(self.energy)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (15, 42))

        # Wave Info
        wave_str = f"WAVE: {self.wave_number} / {self.TOTAL_WAVES}"
        if not self.wave_in_progress and not self.victory:
            wave_str += f" (Next in {int(self.time_until_next_wave / self.FPS) + 1}s)"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 45))

        # Time Dilation Status
        if self.time_dilation_active:
            dilation_text = self.font_small.render("TIME DILATION ACTIVE", True, self.COLOR_TOWER)
            self.screen.blit(dilation_text, (self.CENTER[0] - dilation_text.get_width()//2, self.SCREEN_HEIGHT - 25))

        # Game Over / Victory
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_ENERGY if self.victory else self.COLOR_MESON
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.CENTER[0] - end_text.get_width()//2, self.CENTER[1] - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.nucleus_health,
            "energy": self.energy,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and a display to be available
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Atomic Nucleus Defense - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0    # released
        shift = 0    # released
        
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
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()