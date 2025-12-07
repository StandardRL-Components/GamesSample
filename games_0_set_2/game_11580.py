import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:11:33.797020
# Source Brief: brief_01580.md
# Brief Index: 1580
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Abyssal Genesis: A Gymnasium environment where the agent activates hydrothermal
    vents to cultivate a diverse ecosystem on the ocean floor. The goal is to
    achieve a target biodiversity without causing an ecological collapse from
    runaway algae growth that depletes all oxygen.

    The environment prioritizes visual quality and emergent gameplay, with a dark,
    bioluminescent aesthetic featuring particle effects and procedurally generated life.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Activate hydrothermal vents to cultivate a diverse ecosystem on the ocean floor, "
        "avoiding ecological collapse from runaway algae growth."
    )
    user_guide = "Use the arrow keys to move the cursor and press space to activate a vent."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    BIODIVERSITY_TARGET = 10
    NUM_VENTS = 8
    CURSOR_SPEED = 10
    VENT_ACTIVE_DURATION = 150  # steps
    VENT_RADIUS = 15
    VENT_HEAT_RADIUS = 100

    # --- Colors (Dark, Bioluminescent Theme) ---
    COLOR_BG = (10, 5, 25)
    COLOR_BG_STARS = (50, 40, 80)
    COLOR_VENT_INACTIVE = (100, 80, 200)
    COLOR_VENT_ACTIVE = (220, 200, 255)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    COLOR_HEAT_VIZ = (255, 50, 50)
    COLOR_ALGAE_BLOOM = (0, 100, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 24)
        self.end_font = pygame.font.Font(None, 50)
        self.render_mode = render_mode

        # --- Internal State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.cursor_pos = np.array([0.0, 0.0])
        self.vents = []
        self.lifeforms = []
        self.particles = []
        self.background_stars = []
        self.oxygen_level = 0.0
        self.algae_level = 0.0
        self.known_species = set()
        self.catastrophe_sensitivity = 1.0
        self.space_was_held = False
        self.last_oxygen_level = 100.0
        self.oxygen_warning = False

        self._generate_background()
        # self.reset() is called by the wrapper/user, no need to call it here.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])

        self.vents = self._generate_vents()
        self.lifeforms = []
        self.particles = []

        self.oxygen_level = 100.0
        self.last_oxygen_level = 100.0
        self.algae_level = 0.0

        self.known_species = set()
        self.catastrophe_sensitivity = 1.0 # Resets fragility of ecosystem

        self.space_was_held = False
        self.oxygen_warning = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Unpacking ---
        movement, space_held, _ = action
        space_pressed = (space_held == 1) and not self.space_was_held
        self.space_was_held = (space_held == 1)

        # --- Reward Calculation State ---
        reward_events = {
            "new_lifeform_instance": 0,
            "new_lifeform_species": 0,
            "oxygen_delta": 0.0,
            "oxygen_warning_triggered": False
        }

        # --- Game Logic Updates ---
        self._update_cursor(movement)
        if space_pressed:
            self._handle_vent_activation()

        self._update_vents()
        self._update_particles()
        reward_events = self._update_lifeforms(reward_events)
        self._spawn_new_life(reward_events)
        self._update_environment()

        # --- Calculate Reward ---
        reward = self._calculate_reward(reward_events)
        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win_condition:
                self.score += 100 # Win bonus
            elif terminated: # Only penalize on termination, not truncation
                self.score -= 100 # Lose penalty

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Logic Sub-methods ---

    def _update_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos = np.clip(
            self.cursor_pos, [0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
        )

    def _handle_vent_activation(self):
        for vent in self.vents:
            dist = np.linalg.norm(self.cursor_pos - vent["pos"])
            if dist < self.VENT_RADIUS and not vent["active"]:
                vent["active"] = True
                vent["timer"] = self.VENT_ACTIVE_DURATION
                # Sound: Deep rumble and hiss
                self._create_particles(vent["pos"], 50, self.COLOR_VENT_ACTIVE, 1)
                break

    def _update_vents(self):
        for vent in self.vents:
            if vent["active"]:
                vent["timer"] -= 1
                if vent["timer"] <= 0:
                    vent["active"] = False
                # Emit heat particles periodically
                if self.np_random.random() < 0.3:
                    self._create_particles(vent["pos"], 1, self.COLOR_HEAT_VIZ, 0.5, speed_mult=0.5)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["size"] *= 0.98

    def _update_lifeforms(self, reward_events):
        newly_born = []
        lifeforms_to_remove = []

        for lifeform in self.lifeforms:
            # --- Behavior ---
            lifeform["pos"] += lifeform["vel"]
            lifeform["vel"] += self.np_random.uniform(-0.1, 0.1, 2)
            lifeform["vel"] = np.clip(lifeform["vel"], -1, 1)

            # Bounce off walls
            if not (0 < lifeform["pos"][0] < self.SCREEN_WIDTH): lifeform["vel"][0] *= -1
            if not (0 < lifeform["pos"][1] < self.SCREEN_HEIGHT): lifeform["vel"][1] *= -1
            lifeform["pos"] = np.clip(lifeform["pos"], [0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])

            # --- Metabolism & Energy ---
            lifeform["energy"] -= 0.05 # Base energy cost
            self.oxygen_level -= 0.002 # Respiration

            # Gain energy from nearby active vents
            for vent in self.vents:
                if vent["active"]:
                    dist = np.linalg.norm(lifeform["pos"] - vent["pos"])
                    if dist < self.VENT_HEAT_RADIUS:
                        lifeform["energy"] += 0.5
            lifeform["energy"] = min(lifeform["energy"], 150)

            # --- Reproduction ---
            if lifeform["energy"] > 100:
                lifeform["energy"] = 50
                new_lifeform = lifeform.copy()
                new_lifeform["pos"] = lifeform["pos"] + self.np_random.uniform(-5, 5, 2)
                new_lifeform["energy"] = 50
                newly_born.append(new_lifeform)
                reward_events["new_lifeform_instance"] += 1
                # Sound: soft pop/chime

            # --- Death ---
            if lifeform["energy"] <= 0:
                lifeforms_to_remove.append(lifeform)
                self._create_particles(lifeform["pos"], 10, (100, 100, 100), 0.2) # Death particles
                # Sound: quiet squish

        self.lifeforms = [lf for lf in self.lifeforms if lf not in lifeforms_to_remove]
        self.lifeforms.extend(newly_born)
        return reward_events

    def _spawn_new_life(self, reward_events):
        # New life can only emerge from the primordial soup of an active vent
        for vent in self.vents:
            if vent["active"] and self.np_random.random() < 0.005: # Low chance per step
                species_id = self.np_random.integers(0, 1_000_000)

                is_new_species = species_id not in self.known_species
                if is_new_species:
                    self.known_species.add(species_id)
                    self.catastrophe_sensitivity *= 1.05 # Ecosystem becomes more fragile
                    reward_events["new_lifeform_species"] += 1
                    # Sound: magical discovery chime

                reward_events["new_lifeform_instance"] += 1

                # Procedurally generate species traits
                rng = np.random.default_rng(species_id)
                color = tuple(rng.integers(100, 256, size=3))
                num_segments = rng.integers(1, 5)

                new_lifeform = {
                    "pos": vent["pos"] + self.np_random.uniform(-10, 10, 2),
                    "vel": self.np_random.uniform(-0.5, 0.5, 2),
                    "species_id": species_id,
                    "energy": 50,
                    "color": color,
                    "num_segments": num_segments
                }
                self.lifeforms.append(new_lifeform)
                self._create_particles(new_lifeform["pos"], 30, color, 1.5, speed_mult=2)

    def _update_environment(self):
        # Algae grows faster in high oxygen, and its growth is amplified by ecosystem complexity
        algae_growth = (self.oxygen_level / 100.0) * 0.001 * self.catastrophe_sensitivity
        self.algae_level += algae_growth
        self.algae_level = min(self.algae_level, 1.0)

        # Algae consumes oxygen
        oxygen_consumed_by_algae = self.algae_level * 0.2
        self.oxygen_level -= oxygen_consumed_by_algae

        # Oxygen regenerates very slowly from background processes
        self.oxygen_level += 0.01
        self.oxygen_level = np.clip(self.oxygen_level, 0, 100)

    # --- Reward and Termination ---

    def _calculate_reward(self, events):
        reward = 0
        # Positive rewards for creation
        reward += events["new_lifeform_instance"] * 0.1
        reward += events["new_lifeform_species"] * 1.0

        # Negative reward for environmental degradation
        oxygen_delta = self.oxygen_level - self.last_oxygen_level
        if oxygen_delta < 0:
            reward += oxygen_delta * 0.1 # e.g., -2 drop -> -0.2 reward

        # Event-based penalty for near-catastrophe
        if self.oxygen_level < 20 and not self.oxygen_warning:
            reward -= 5.0
            self.oxygen_warning = True
        elif self.oxygen_level >= 20:
            self.oxygen_warning = False # Reset warning

        self.last_oxygen_level = self.oxygen_level
        return reward

    def _check_termination(self):
        # Win condition
        if len(self.known_species) >= self.BIODIVERSITY_TARGET:
            self.win_condition = True
            return True
        # Loss conditions
        if self.oxygen_level <= 0:
            return True # Catastrophe
        # Extinction: if life has existed before, and now it's all gone.
        if self.steps > 50 and not self.lifeforms and len(self.known_species) > 0:
            return True # Extinction
        return False

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_environment_effects()
        self._render_vents()
        self._render_particles()
        self._render_lifeforms()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.background_stars:
            pygame.draw.circle(self.screen, self.COLOR_BG_STARS, star['pos'], star['size'])

    def _render_environment_effects(self):
        if self.algae_level > 0.01:
            overlay = self.screen.copy()
            overlay.fill(self.COLOR_ALGAE_BLOOM)
            overlay.set_alpha(int(self.algae_level * 100))
            self.screen.blit(overlay, (0, 0))

    def _render_vents(self):
        for vent in self.vents:
            pos = tuple(map(int, vent["pos"]))
            color = self.COLOR_VENT_ACTIVE if vent["active"] else self.COLOR_VENT_INACTIVE

            # Glow effect
            if vent["active"]:
                for i in range(4):
                    alpha = 100 - i * 25
                    radius = self.VENT_RADIUS + i * 4
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, alpha))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.VENT_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.VENT_RADIUS, color)

    def _render_lifeforms(self):
        for lifeform in self.lifeforms:
            pos = lifeform["pos"]
            color = lifeform["color"]
            # Render body as a series of circles
            vel_norm = np.linalg.norm(lifeform["vel"])
            direction = lifeform["vel"] / (vel_norm + 1e-6)
            for i in range(lifeform["num_segments"]):
                offset = direction * i * -3
                segment_pos = tuple(map(int, pos + offset))
                size = max(2, int(6 - i * 1.5))
                pygame.gfxdraw.filled_circle(self.screen, segment_pos[0], segment_pos[1], size, color)
                pygame.gfxdraw.aacircle(self.screen, segment_pos[0], segment_pos[1], size, color)

    def _render_particles(self):
        for p in self.particles:
            pos = tuple(map(int, p["pos"]))
            size = int(p["size"])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p["color"])

    def _render_cursor(self):
        if self.game_over: return
        pos = tuple(map(int, self.cursor_pos))
        r = 10
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0] - r, pos[1]), (pos[0] + r, pos[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - r), (pos[0], pos[1] + r), 1)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(text_surf, pos)

        # Helper to draw a bar
        def draw_bar(pos, size, label, value, max_value, color, critical_color):
            bar_color = critical_color if value / max_value < 0.2 else color
            fill_width = int(size[0] * (value / max_value))
            pygame.draw.rect(self.screen, (50, 50, 50), (*pos, *size), 1)
            if fill_width > 0:
                pygame.draw.rect(self.screen, bar_color, (pos[0], pos[1], fill_width, size[1]))
            draw_text(label, (pos[0] + 5, pos[1]), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # UI Elements
        draw_bar((10, 10), (150, 20), "Oxygen", self.oxygen_level, 100, (0, 150, 255), (255, 0, 0))
        draw_bar((170, 10), (150, 20), "Algae", self.algae_level, 1.0, (0, 200, 50), (150, 200, 0))

        bio_text = f"Biodiversity: {len(self.known_species)} / {self.BIODIVERSITY_TARGET}"
        draw_text(bio_text, (330, 12), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        score_text = f"Score: {int(self.score)}"
        draw_text(score_text, (520, 12), self.ui_font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "TERRAFORMING COMPLETE" if self.win_condition else "ECOLOGICAL COLLAPSE"
        color = (100, 255, 100) if self.win_condition else (255, 100, 100)

        text_surf = self.end_font.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    # --- Helper and Info Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "oxygen": self.oxygen_level,
            "algae": self.algae_level,
            "biodiversity": len(self.known_species),
            "lifeforms": len(self.lifeforms)
        }

    def _generate_vents(self):
        vents = []
        for _ in range(self.NUM_VENTS):
            vents.append({
                "pos": self.np_random.uniform(
                    [self.SCREEN_WIDTH * 0.1, self.SCREEN_HEIGHT * 0.2],
                    [self.SCREEN_WIDTH * 0.9, self.SCREEN_HEIGHT * 0.9]
                ),
                "active": False,
                "timer": 0
            })
        return vents

    def _create_particles(self, pos, count, color, lifespan_mult=1.0, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.uniform(20, 50) * lifespan_mult,
                "size": self.np_random.uniform(2, 5),
                "color": color
            })

    def _generate_background(self):
        self.background_stars = []
        for _ in range(100):
            self.background_stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'size': random.randint(1, 2)
            })

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    # This block will only run if the script is executed directly
    # It will not run when the class is imported by the test suite.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play ---
    obs, info = env.reset()
    done = False

    # Setup a window to see the rendering
    pygame.display.set_caption("Abyssal Genesis - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Press Q to quit.")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        keys = pygame.key.get_pressed()

        # Map keys to MultiDiscrete action
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Biodiversity: {info['biodiversity']}")

        if done:
            print("\n--- Game Over ---")
            print(f"Final Score: {info['score']:.2f}")
            print(f"Reason: {'Win' if env.win_condition else 'Loss'}")
            # Keep window open for a bit to see end screen
            pygame.time.wait(3000)

        clock.tick(30) # Limit frame rate for playability

    env.close()