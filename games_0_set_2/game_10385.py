import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: Defend planets by firing rhythmic energy pulses through portals
    to trigger chain reactions and destroy invading enemy formations.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Portal Selection (0=none, 1=TL, 2=TR, 3=BL, 4=BR)
    - actions[1]: Fire Button (0=released, 1=held/charging)
    - actions[2]: Unused (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB Array of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your planet by firing energy pulses from four portals to destroy waves of orbiting enemies."
    user_guide = "Use number keys 1-4 to select a portal (1=TL, 2=TR, 3=BL, 4=BR). Hold space to charge a pulse and release to fire."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000 # Increased from 1000 to allow for longer games
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLANET_MAIN = (20, 40, 80)
    COLOR_PLANET_RING = (30, 60, 120)
    COLOR_PORTAL_INACTIVE = (0, 100, 200)
    COLOR_PORTAL_ACTIVE = (100, 200, 255)
    COLOR_PULSE = (255, 255, 0)
    COLOR_ENEMY_A = (255, 0, 100)
    COLOR_ENEMY_B = (220, 50, 220)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)

    # Game Parameters
    PLANET_RADIUS = 70
    PLANET_MAX_HEALTH = 100
    PORTAL_RADIUS = 15
    PULSE_BASE_SPEED = 6
    PULSE_MAX_CHARGE = 1.0
    PULSE_CHARGE_RATE = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.planet_health = 0
        self.wave = 0
        self.selected_portal_idx = 0
        self.charge_level = 0.0
        self.space_was_held = False

        self.center_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)

        margin_x = 100
        margin_y = 70
        self.portals = [
            pygame.Vector2(margin_x, margin_y),
            pygame.Vector2(self.WIDTH - margin_x, margin_y),
            pygame.Vector2(margin_x, self.HEIGHT - margin_y),
            pygame.Vector2(self.WIDTH - margin_x, self.HEIGHT - margin_y),
        ]

        self.enemies = []
        self.pulses = []
        self.particles = []

        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.planet_health = self.PLANET_MAX_HEALTH
        self.wave = 1
        self.selected_portal_idx = 0
        self.charge_level = 0.0
        self.space_was_held = False

        self.enemies.clear()
        self.pulses.clear()
        self.particles.clear()

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1

        step_reward = 0.0

        # --- 1. Handle Actions ---
        portal_selection = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        if portal_selection in [1, 2, 3, 4]:
            self.selected_portal_idx = portal_selection - 1

        self._update_charge(space_held)

        # Fire pulse on release
        if not space_held and self.space_was_held:
            self._fire_pulse() # Sound: PulseFire.wav
        self.space_was_held = space_held

        # --- 2. Update Game Logic ---
        reward_from_updates = self._update_game_state()
        step_reward += reward_from_updates

        # --- 3. Check for Wave Clear ---
        if not self.enemies and not self.game_over:
            step_reward += 10.0 # Wave clear bonus
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                self.won = True
                self.game_over = True
            else:
                self._spawn_wave()

        # --- 4. Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.won:
                step_reward += 100.0 # Win bonus
            elif self.planet_health <= 0:
                step_reward -= 100.0 # Loss penalty

        # Make sure terminated and truncated are not both true
        if truncated:
            terminated = False

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_charge(self, space_held):
        if space_held:
            self.charge_level = min(self.PULSE_MAX_CHARGE, self.charge_level + self.PULSE_CHARGE_RATE)
        else:
            self.charge_level = 0.0

    def _fire_pulse(self):
        portal_pos = self.portals[self.selected_portal_idx]
        direction = (self.center_pos - portal_pos).normalize()

        # Pulses are more powerful when charged
        speed = self.PULSE_BASE_SPEED + 3 * self.charge_level
        radius = 5 + 10 * self.charge_level

        self.pulses.append({
            "pos": portal_pos.copy(),
            "vel": direction * speed,
            "radius": radius,
            "lifespan": 150 # frames
        })
        self._create_particles(portal_pos, 15, self.COLOR_PULSE, 2.0)

    def _update_game_state(self):
        update_reward = 0.0

        # Update Pulses
        for pulse in self.pulses[:]:
            pulse["pos"] += pulse["vel"]
            pulse["lifespan"] -= 1
            if pulse["lifespan"] <= 0 or not self.screen.get_rect().collidepoint(pulse["pos"]):
                self.pulses.remove(pulse)
                update_reward -= 0.1 # Penalty for missing
                continue

        # Update Enemies
        for enemy in self.enemies[:]:
            enemy["move_timer"] += 1
            # Simple orbiting pattern
            angle = enemy["start_angle"] + enemy["move_timer"] * enemy["speed"]
            enemy["pos"].x = self.center_pos.x + enemy["orbit_radius"] * math.cos(angle)
            enemy["pos"].y = self.center_pos.y + enemy["orbit_radius"] * math.sin(angle)

            # Check collision with planet
            if enemy["pos"].distance_to(self.center_pos) < self.PLANET_RADIUS + enemy["radius"]:
                self.planet_health = max(0, self.planet_health - enemy["damage"])
                self._create_particles(enemy["pos"], 30, self.COLOR_ENEMY_A, 3.0)
                self.enemies.remove(enemy) # Sound: PlanetHit.wav
                if self.planet_health <= 0:
                    self.game_over = True

        # Update Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Check Pulse-Enemy Collisions
        for pulse in self.pulses[:]:
            for enemy in self.enemies[:]:
                if pulse["pos"].distance_to(enemy["pos"]) < pulse["radius"] + enemy["radius"]:
                    # Sound: EnemyHit.wav
                    self._create_particles(enemy["pos"], 20, self.COLOR_PULSE, 2.5)
                    enemy["hp"] -= 1
                    update_reward += 1.0 # Reward for hitting
                    if pulse in self.pulses: self.pulses.remove(pulse)

                    if enemy["hp"] <= 0:
                        # Sound: EnemyExplode.wav
                        self._create_particles(enemy["pos"], 50, self.COLOR_ENEMY_B, 4.0)
                        self.enemies.remove(enemy)
                        self.score += 10
                        update_reward += 5.0 # Reward for destroying
                    break

        return update_reward

    def _check_termination(self):
        if self.game_over:
            return True
        # Time limit is handled by truncation, not termination
        return False

    def _spawn_wave(self):
        num_enemies = 3 + (self.wave - 1) // 2
        for i in range(num_enemies):
            orbit_radius = self.PLANET_RADIUS + 50 + self.np_random.uniform(20, 80)
            start_angle = self.np_random.uniform(0, 2 * math.pi)
            speed = 0.01 + self.wave * 0.005 * self.np_random.uniform(0.8, 1.2)

            self.enemies.append({
                "pos": pygame.Vector2(0, 0),
                "hp": 1 + self.wave,
                "radius": 12,
                "damage": 10 + self.wave,
                "orbit_radius": orbit_radius,
                "start_angle": start_angle,
                "speed": speed,
                "move_timer": 0,
            })

    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed)),
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "health": self.planet_health}

    def _render_game(self):
        # Planet
        pygame.gfxdraw.filled_circle(self.screen, int(self.center_pos.x), int(self.center_pos.y), self.PLANET_RADIUS, self.COLOR_PLANET_MAIN)
        pygame.gfxdraw.aacircle(self.screen, int(self.center_pos.x), int(self.center_pos.y), self.PLANET_RADIUS, self.COLOR_PLANET_RING)
        pygame.gfxdraw.aacircle(self.screen, int(self.center_pos.x), int(self.center_pos.y), self.PLANET_RADIUS - 20, self.COLOR_PLANET_RING)

        # Portals
        for i, pos in enumerate(self.portals):
            is_selected = (i == self.selected_portal_idx)
            color = self.COLOR_PORTAL_ACTIVE if is_selected else self.COLOR_PORTAL_INACTIVE

            # Glow effect
            glow_radius = int(self.PORTAL_RADIUS * (1.5 if is_selected else 1.2))
            glow_color = (*color, 60)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.PORTAL_RADIUS, color)

            # Charge indicator
            if is_selected and self.charge_level > 0:
                charge_radius = int(self.PORTAL_RADIUS * (1 + self.charge_level * 0.8))
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), charge_radius, self.COLOR_PULSE)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], max(0, min(255, alpha)))
            try: # Use a try-except block for cases where alpha might be invalid for a color
                pygame.draw.circle(self.screen, color, (int(p["pos"].x), int(p["pos"].y)), int(p["radius"]))
            except (ValueError, TypeError):
                pass

        # Pulses
        for pulse in self.pulses:
            r, g, b = self.COLOR_PULSE
            pygame.gfxdraw.filled_circle(self.screen, int(pulse["pos"].x), int(pulse["pos"].y), int(pulse["radius"]), (r,g,b,150))
            pygame.gfxdraw.aacircle(self.screen, int(pulse["pos"].x), int(pulse["pos"].y), int(pulse["radius"]), self.COLOR_PULSE)

        # Enemies
        for enemy in self.enemies:
            # Glow
            glow_radius = int(enemy["radius"] * 1.8)
            glow_color = (*self.COLOR_ENEMY_A, 80)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (int(enemy["pos"].x - glow_radius), int(enemy["pos"].y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(enemy["pos"].x), int(enemy["pos"].y), enemy["radius"], self.COLOR_ENEMY_A)
            pygame.gfxdraw.aacircle(self.screen, int(enemy["pos"].x), int(enemy["pos"].y), enemy["radius"], self.COLOR_ENEMY_B)

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Planet Health Bar
        health_bar_width = 200
        health_bar_height = 15
        health_bar_x = self.center_pos.x - health_bar_width / 2
        health_bar_y = 20

        health_ratio = self.planet_health / self.PLANET_MAX_HEALTH
        current_health_width = health_bar_width * health_ratio

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), border_radius=4)
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (health_bar_x, health_bar_y, current_health_width, health_bar_height), border_radius=4)

        # Game Over / Win Text
        if self.game_over:
            text_str = "VICTORY" if self.won else "PLANET LOST"
            color = self.COLOR_HEALTH_BAR if self.won else self.COLOR_ENEMY_A
            game_over_text = self.font_game_over.render(text_str, True, color)
            text_rect = game_over_text.get_rect(center=self.center_pos)
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you might need to comment out the os.environ line at the top
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Planet Defender")
    clock = pygame.time.Clock()

    total_reward = 0

    # Mapping keys to actions
    key_map = {
        pygame.K_1: 1, pygame.K_KP_1: 1, # Portal TL
        pygame.K_2: 2, pygame.K_KP_2: 2, # Portal TR
        pygame.K_3: 3, pygame.K_KP_3: 3, # Portal BL
        pygame.K_4: 4, pygame.K_KP_4: 4, # Portal BR
    }

    action = [0, 0, 0] # [portal, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    action[0] = key_map[event.key]
                else:
                    action[0] = 0 # No change on other keys
            if event.type == pygame.KEYUP:
                 action[0] = 0 # Stop changing portal on key up

        keys = pygame.key.get_pressed()
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()