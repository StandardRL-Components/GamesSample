import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your factory from waves of incoming enemies in this top-down shooter. "
        "Pilot one of two unique mechs, switching between them to adapt to the threat."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your mech. "
        "Press space to fire at the nearest enemy. Press shift to switch between your mechs."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 20

        # --- Colors (Steampunk/Industrial Theme) ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_BG_ACCENT = (30, 35, 40)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ENEMY_A = (255, 50, 50)
        self.COLOR_ENEMY_B = (255, 150, 50)
        self.COLOR_PROJECTILE_PLAYER = (100, 255, 255)
        self.COLOR_PROJECTILE_ENEMY = (255, 180, 100)
        self.COLOR_FACTORY = (100, 110, 120)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        self.COLOR_WAVE_TEXT = (255, 200, 0)

        # --- Action Space ---
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Observation Space ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_wave = pygame.font.SysFont("Impact", 48)

        # --- State variables initialized in reset() ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.wave_number = None
        self.factory_health = None
        self.max_factory_health = None
        self.player_fighters = None
        self.selected_fighter_idx = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.game_phase = None
        self.wave_transition_timer = None
        self.last_shift_state = None
        self.last_space_state = None
        self.step_reward = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        self.max_factory_health = 1000
        self.factory_health = self.max_factory_health

        # Create player fighters (pre-defined builds)
        self.player_fighters = [
            self._create_fighter("scout", [self.WIDTH * 0.15, self.HEIGHT * 0.3]),
            self._create_fighter("tank", [self.WIDTH * 0.15, self.HEIGHT * 0.7]),
        ]
        self.selected_fighter_idx = 0

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.game_phase = "COMBAT"
        self.wave_transition_timer = 0
        self.last_shift_state = 0
        self.last_space_state = 0

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0

        self._handle_input(action)
        self._update_physics()
        self._handle_collisions()
        self._update_game_state()
        self._cleanup_entities()

        self.steps += 1
        self.score += self.step_reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Fighter Switching (on press) ---
        if shift_held and not self.last_shift_state:
            # SFX: UI_Switch.wav
            self.selected_fighter_idx = (self.selected_fighter_idx + 1) % len(
                self.player_fighters
            )
            # Add particles to indicate switch
            if self.player_fighters:
                pos = self.player_fighters[self.selected_fighter_idx]["pos"]
                self._create_particles(pos, 20, self.COLOR_PLAYER, 1.5, 30)
        self.last_shift_state = shift_held

        if not self.player_fighters:
            return

        selected_fighter = self.player_fighters[self.selected_fighter_idx]

        # --- Movement ---
        force_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}
        if movement in force_map:
            force_dir = force_map[movement]
            selected_fighter["vel"][0] += force_dir[0] * selected_fighter["thrust"]
            selected_fighter["vel"][1] += force_dir[1] * selected_fighter["thrust"]

        # --- Shooting (on press) ---
        if (
            space_held
            and not self.last_space_state
            and selected_fighter["shoot_cooldown"] <= 0
        ):
            # SFX: Laser_Shoot.wav
            selected_fighter["shoot_cooldown"] = selected_fighter["fire_rate"]

            start_pos = list(selected_fighter["pos"])
            # Aim at the closest enemy
            closest_enemy = None
            min_dist = float("inf")
            for enemy in self.enemies:
                dist = math.hypot(
                    enemy["pos"][0] - start_pos[0], enemy["pos"][1] - start_pos[1]
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy

            if closest_enemy:
                target_pos = closest_enemy["pos"]
                angle = math.atan2(
                    target_pos[1] - start_pos[1], target_pos[0] - start_pos[0]
                )
                vel = [math.cos(angle) * 10, math.sin(angle) * 10]
                self.projectiles.append(
                    {
                        "pos": start_pos,
                        "vel": vel,
                        "radius": 4,
                        "damage": selected_fighter["damage"],
                        "owner": "player",
                        "color": self.COLOR_PROJECTILE_PLAYER,
                        "lifespan": 60,
                    }
                )
                # Muzzle flash
                flash_pos = [start_pos[0] + vel[0] * 2, start_pos[1] + vel[1] * 2]
                self._create_particles(flash_pos, 5, (255, 255, 255), 2.0, 5)

        self.last_space_state = space_held

    def _update_physics(self):
        # Update Fighters
        for fighter in self.player_fighters:
            fighter["vel"][0] *= 0.9  # Friction
            fighter["vel"][1] *= 0.9
            fighter["pos"][0] += fighter["vel"][0]
            fighter["pos"][1] += fighter["vel"][1]
            fighter["shoot_cooldown"] = max(0, fighter["shoot_cooldown"] - 1)
            # Boundary collision
            fighter["pos"][0] = np.clip(
                fighter["pos"][0], fighter["radius"], self.WIDTH - fighter["radius"]
            )
            fighter["pos"][1] = np.clip(
                fighter["pos"][1], fighter["radius"], self.HEIGHT - fighter["radius"]
            )

        # Update Enemies
        for enemy in self.enemies:
            # AI: Move towards the factory (left side)
            target_y = self.HEIGHT / 2  # Simple target
            angle = math.atan2(target_y - enemy["pos"][1], 0 - enemy["pos"][0])
            base_speed = enemy["speed"] * (1 + (self.wave_number - 1) * 0.02)

            enemy["vel"][0] = math.cos(angle) * base_speed
            enemy["vel"][1] = math.sin(angle) * base_speed

            # Add oscillation for 'bomber' type
            if enemy["type"] == "bomber":
                oscillation = math.sin(self.steps * 0.1 + enemy["id"]) * base_speed * 1.5
                enemy["vel"][0] += math.sin(angle) * oscillation
                enemy["vel"][1] += -math.cos(angle) * oscillation

            enemy["pos"][0] += enemy["vel"][0]
            enemy["pos"][1] += enemy["vel"][1]

        # Update Projectiles
        for p in self.projectiles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

        # Update Particles
        for part in self.particles:
            part["pos"][0] += part["vel"][0]
            part["pos"][1] += part["vel"][1]
            part["vel"][1] += 0.05  # Gravity
            part["lifespan"] -= 1
            part["radius"] = max(0, part["radius"] * 0.95)

    def _handle_collisions(self):
        # Player Projectiles vs Enemies
        for p in self.projectiles[:]:
            if p["owner"] != "player":
                continue
            for enemy in self.enemies[:]:
                dist = math.hypot(
                    p["pos"][0] - enemy["pos"][0], p["pos"][1] - enemy["pos"][1]
                )
                if dist < p["radius"] + enemy["radius"]:
                    # SFX: Hit_Confirm.wav
                    enemy["health"] -= p["damage"]
                    self.step_reward += 0.1
                    self._create_particles(p["pos"], 10, self.COLOR_ENEMY_A, 1.0, 20)
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    if enemy["health"] <= 0:
                        # SFX: Explosion.wav
                        self.step_reward += 1.0
                        self._create_particles(
                            enemy["pos"], 50, self.COLOR_ENEMY_B, 3.0, 40
                        )
                        self.enemies.remove(enemy)
                    break

        # Enemies vs Player Fighters (Contact Damage)
        for enemy in self.enemies:
            for fighter in self.player_fighters[:]:
                dist = math.hypot(
                    enemy["pos"][0] - fighter["pos"][0],
                    enemy["pos"][1] - fighter["pos"][1],
                )
                if dist < enemy["radius"] + fighter["radius"]:
                    # SFX: Player_Hit.wav
                    fighter["health"] -= 1  # Contact damage
                    enemy["health"] -= 5  # Ramming damage to enemy
                    self.step_reward -= 0.1
                    self._create_particles(fighter["pos"], 5, self.COLOR_PLAYER, 1.0, 15)
                    if fighter["health"] <= 0:
                        self.step_reward -= 5.0
                        self._create_particles(
                            fighter["pos"], 50, self.COLOR_PLAYER, 3.0, 40
                        )
                        self.player_fighters.remove(fighter)
                        self.selected_fighter_idx = 0 if self.player_fighters else -1
                    if enemy["health"] <= 0 and enemy in self.enemies:
                        self.step_reward += 1.0
                        self._create_particles(
                            enemy["pos"], 50, self.COLOR_ENEMY_B, 3.0, 40
                        )
                        self.enemies.remove(enemy)

        # Enemies vs Factory
        for enemy in self.enemies[:]:
            if enemy["pos"][0] < 40 + enemy["radius"]:  # Factory area
                # SFX: Factory_Damage.wav
                self.factory_health -= 10
                self.step_reward -= 0.5  # Penalty for factory damage
                self._create_particles(enemy["pos"], 30, self.COLOR_FACTORY, 2.0, 30)
                self.enemies.remove(enemy)

    def _update_game_state(self):
        # Wave transition
        if (
            not self.enemies
            and self.game_phase == "COMBAT"
            and self.wave_number < self.MAX_WAVES
        ):
            self.game_phase = "WAVE_TRANSITION"
            self.wave_transition_timer = self.FPS * 3  # 3 second pause
            self.step_reward += 10.0

        if self.game_phase == "WAVE_TRANSITION":
            self.wave_transition_timer -= 1
            if self.wave_transition_timer <= 0:
                self.wave_number += 1
                self._spawn_wave()
                self.game_phase = "COMBAT"

    def _cleanup_entities(self):
        self.projectiles = [
            p
            for p in self.projectiles
            if p["lifespan"] > 0
            and 0 < p["pos"][0] < self.WIDTH
            and 0 < p["pos"][1] < self.HEIGHT
        ]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _check_termination(self):
        if self.factory_health <= 0:
            self.game_over = True
            self.step_reward -= 100.0
            return True
        if not self.player_fighters:
            self.game_over = True
            return True
        if self.wave_number >= self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.step_reward += 100.0
            return True
        return False

    def _spawn_wave(self):
        num_grunts = 2 + self.wave_number
        num_bombers = self.wave_number // 5

        base_health = 10 * (1 + (self.wave_number - 1) * 0.1)

        for i in range(num_grunts):
            self.enemies.append(self._create_enemy("grunt", base_health))
        for i in range(num_bombers):
            self.enemies.append(self._create_enemy("bomber", base_health * 1.5))

    def _create_fighter(self, f_type, pos):
        if f_type == "scout":
            return {
                "pos": pos,
                "vel": [0, 0],
                "radius": 15,
                "health": 75,
                "max_health": 75,
                "thrust": 0.8,
                "fire_rate": 10,
                "damage": 8,
                "shoot_cooldown": 0,
                "type": f_type,
            }
        elif f_type == "tank":
            return {
                "pos": pos,
                "vel": [0, 0],
                "radius": 20,
                "health": 150,
                "max_health": 150,
                "thrust": 0.4,
                "fire_rate": 25,
                "damage": 25,
                "shoot_cooldown": 0,
                "type": f_type,
            }

    def _create_enemy(self, e_type, health):
        pos = [
            self.WIDTH + random.randint(20, 100),
            random.randint(20, self.HEIGHT - 20),
        ]
        if e_type == "grunt":
            return {
                "id": random.random(),
                "pos": pos,
                "vel": [0, 0],
                "radius": 12,
                "health": health,
                "max_health": health,
                "speed": 1.5,
                "type": e_type,
                "color": self.COLOR_ENEMY_A,
            }
        elif e_type == "bomber":
            return {
                "id": random.random(),
                "pos": pos,
                "vel": [0, 0],
                "radius": 16,
                "health": health,
                "max_health": health,
                "speed": 1.0,
                "type": e_type,
                "color": self.COLOR_ENEMY_B,
            }

    def _create_particles(self, pos, count, color, speed_mult, lifespan):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(
                {
                    "pos": list(pos),
                    "vel": vel,
                    "radius": random.uniform(2, 5),
                    "lifespan": random.randint(lifespan // 2, lifespan),
                    "color": color,
                }
            )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "factory_health": self.factory_health,
            "enemies_left": len(self.enemies),
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_ACCENT, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_ACCENT, (0, i), (self.WIDTH, i), 1)

    def _render_game(self):
        # Render Factory
        pygame.draw.rect(self.screen, self.COLOR_FACTORY, (0, 0, 40, self.HEIGHT))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 40, self.HEIGHT), 2)

        # Render Particles
        for part in self.particles:
            pygame.draw.circle(
                self.screen, part["color"], [int(p) for p in part["pos"]], int(part["radius"])
            )

        # Render Projectiles
        for p in self.projectiles:
            pos = [int(val) for val in p["pos"]]
            self._draw_glowing_circle(self.screen, p["color"], pos, p["radius"], 10)

        # Render Enemies
        for enemy in self.enemies:
            self._render_entity(enemy, enemy["color"])

        # Render Fighters
        for i, fighter in enumerate(self.player_fighters):
            is_selected = i == self.selected_fighter_idx
            self._render_entity(fighter, self.COLOR_PLAYER, is_selected)

    def _render_entity(self, entity, color, is_selected=False):
        pos = [int(p) for p in entity["pos"]]
        radius = entity["radius"]

        if is_selected:
            self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, pos, radius, 30)

        # Body
        pygame.draw.circle(self.screen, color, pos, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)  # Outline

        # Health bar
        if entity["health"] < entity["max_health"]:
            bar_width = radius * 1.5
            bar_height = 5
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - radius - 10
            health_ratio = max(0, entity["health"] / entity["max_health"])
            pygame.draw.rect(
                self.screen,
                self.COLOR_HEALTH_BAR_BG,
                (bar_x, bar_y, bar_width, bar_height),
            )
            pygame.draw.rect(
                self.screen,
                self.COLOR_HEALTH_BAR,
                (bar_x, bar_y, bar_width * health_ratio, bar_height),
            )

    def _draw_glowing_circle(self, surface, color, center, radius, glow_radius):
        for i in range(glow_radius, 0, -2):
            alpha = int(255 * (1 - i / glow_radius))
            pygame.gfxdraw.filled_circle(
                surface, center[0], center[1], int(radius + i), color + (alpha // 8,)
            )
        pygame.gfxdraw.aacircle(surface, center[0], center[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(radius), color)

    def _render_ui(self):
        # Factory Health
        factory_health_ratio = max(0, self.factory_health / self.max_factory_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (50, 15, 200, 20))
        pygame.draw.rect(
            self.screen, self.COLOR_HEALTH_BAR, (50, 15, 200 * factory_health_ratio, 20)
        )
        ui_text = self.font_ui.render("FACTORY", True, self.COLOR_UI_TEXT)
        self.screen.blit(ui_text, (55, 16))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 150, 15))

        # Wave transition text
        if self.game_phase == "WAVE_TRANSITION":
            wave_text_surf = self.font_wave.render(
                f"WAVE {self.wave_number} COMPLETE", True, self.COLOR_WAVE_TEXT
            )
            text_rect = wave_text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 30))
            self.screen.blit(wave_text_surf, text_rect)

            next_wave_timer = math.ceil(self.wave_transition_timer / self.FPS)
            next_wave_surf = self.font_ui.render(
                f"Next wave in {next_wave_timer}...", True, self.COLOR_UI_TEXT
            )
            next_rect = next_wave_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(next_wave_surf, next_rect)
        else:
            wave_text = self.font_ui.render(
                f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT
            )
            self.screen.blit(wave_text, (self.WIDTH / 2 - 50, 15))

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Mech Factory Defense")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0

        while running:
            movement = 0  # 0=none
            space_held = 0  # 0=released
            shift_held = 0  # 0=released

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                movement = 4

            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
                # Wait a moment before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("This is expected in a headless environment. The environment code is likely correct.")