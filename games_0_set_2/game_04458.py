import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to fire. Press shift to rotate your aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a powerful robot in a top-down arena, blasting enemies and collecting power-ups to achieve total annihilation."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.TOTAL_ENEMIES_TO_DEFEAT = 20
        self.PLAYER_SIZE = 12
        self.ENEMY_SIZE = 10
        self.POWERUP_SIZE = 8
        self.PLAYER_MAX_HEALTH = 100
        self.ENEMY_HEALTH = 3
        self.PLAYER_SHOOT_COOLDOWN = 6  # frames
        self.ENEMY_SHOOT_COOLDOWN = 45  # frames
        self.POWERUP_SPAWN_CHANCE = 0.01
        self.POWERUP_DURATION = 300  # frames

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 80, 80, 50)
        self.COLOR_ENEMY = (80, 120, 255)
        self.COLOR_ENEMY_GLOW = (80, 120, 255, 50)
        self.COLOR_PLAYER_PROJ = (200, 255, 255)
        self.COLOR_ENEMY_PROJ = (255, 150, 200)
        self.COLOR_HEALTH_POWERUP = (80, 255, 80)
        self.COLOR_SPEED_POWERUP = (255, 255, 80)
        self.COLOR_DAMAGE_POWERUP = (200, 80, 255)
        self.COLOR_EXPLOSION = (255, 180, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (80, 220, 80)
        self.COLOR_HEALTH_BAR_BG = (120, 40, 40)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 20, bold=True)

        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        self.dead_enemies_timers = []
        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        self.prev_shift_held = False
        self.rng = None

        # self.validate_implementation() # Commented out for submission, but useful for dev

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        self.prev_shift_held = False

        self.player = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "health": self.PLAYER_MAX_HEALTH,
            "aim_angle": 0,
            "shoot_cooldown": 0,
            "base_speed": 3,
            "base_damage": 1,
            "speed_boost_timer": 0,
            "damage_boost_timer": 0,
        }

        self.enemies = []
        self.dead_enemies_timers = []
        for _ in range(5):  # Start with 5 enemies
            self._spawn_enemy()

        self.projectiles = []
        self.powerups = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Update Game Logic ---
        self._handle_input(movement, space_held, shift_held)

        self._update_player()
        reward += self._update_enemies()
        reward += self._update_projectiles()
        reward += self._update_powerups()
        self._update_particles()
        self._update_spawners()

        self.steps += 1

        terminated = self._check_termination()
        if terminated:
            if self.player["health"] <= 0:
                reward -= 100
            elif self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT:
                reward += 100

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:
            move_vec.y = -1  # Up
        elif movement == 2:
            move_vec.y = 1  # Down
        elif movement == 3:
            move_vec.x = -1  # Left
        elif movement == 4:
            move_vec.x = 1  # Right

        speed = self.player["base_speed"] * (
            1.5 if self.player["speed_boost_timer"] > 0 else 1
        )
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["pos"] += move_vec * speed
            # Thruster particles
            if self.steps % 2 == 0:
                self._create_particle(
                    self.player["pos"] - move_vec * self.PLAYER_SIZE,
                    -move_vec * 2,
                    10,
                    self.COLOR_ENEMY_PROJ,
                    2,
                )

        # Aiming (on key press, not hold)
        if shift_held and not self.prev_shift_held:
            self.player["aim_angle"] = (self.player["aim_angle"] + 45) % 360
        self.prev_shift_held = shift_held

        # Shooting
        if space_held and self.player["shoot_cooldown"] == 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            aim_vec = pygame.Vector2(1, 0).rotate(-self.player["aim_angle"])
            proj_pos = self.player["pos"] + aim_vec * (self.PLAYER_SIZE + 5)
            self.projectiles.append(
                {
                    "pos": proj_pos,
                    "vel": aim_vec * 10,
                    "owner": "player",
                    "damage": self.player["base_damage"]
                    * (2 if self.player["damage_boost_timer"] > 0 else 1),
                }
            )
            # sfx: player_shoot.wav
            # Muzzle flash
            self._create_particle(
                proj_pos,
                pygame.Vector2(0, 0),
                5,
                self.COLOR_PLAYER_PROJ,
                3,
                grow=True,
            )

    def _update_player(self):
        # Cooldowns and boosts
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        if self.player["speed_boost_timer"] > 0:
            self.player["speed_boost_timer"] -= 1
        if self.player["damage_boost_timer"] > 0:
            self.player["damage_boost_timer"] -= 1

        # Clamp position to screen bounds
        self.player["pos"].x = np.clip(
            self.player["pos"].x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE
        )
        self.player["pos"].y = np.clip(
            self.player["pos"].y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE
        )

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies:
            # Movement
            direction_to_player = self.player["pos"] - enemy["pos"]
            if direction_to_player.length() > 0:
                direction_to_player.normalize_ip()
                enemy["pos"] += direction_to_player * enemy["speed"]

            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0 and direction_to_player.length() < 250:
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN + self.rng.integers(
                    -10, 10
                )

                difficulty_speed_mult = 1.0 + (self.steps / 200) * 0.05
                proj_speed = 4 * difficulty_speed_mult

                self.projectiles.append(
                    {
                        "pos": pygame.Vector2(enemy["pos"]),
                        "vel": direction_to_player * proj_speed,
                        "owner": "enemy",
                        "damage": 10,
                    }
                )
                # sfx: enemy_shoot.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"]

            # Check OOB
            if not (0 < proj["pos"].x < self.WIDTH and 0 < proj["pos"].y < self.HEIGHT):
                continue

            collided = False
            # Player projectile collision
            if proj["owner"] == "player":
                for enemy in self.enemies:
                    if (proj["pos"] - enemy["pos"]).length() < self.ENEMY_SIZE:
                        enemy["health"] -= proj["damage"]
                        collided = True
                        reward += 0.1  # Hit reward
                        # sfx: enemy_hit.wav
                        self._create_hit_spark(proj["pos"], self.COLOR_ENEMY)
                        if enemy["health"] <= 0:
                            reward += 1.0  # Kill reward
                            self.score += 10
                            self.enemies_defeated += 1
                            self._create_explosion(
                                enemy["pos"], self.COLOR_EXPLOSION, 15
                            )
                            self.enemies.remove(enemy)
                            self.dead_enemies_timers.append(30)  # Respawn timer
                            # sfx: explosion.wav
                        break
            # Enemy projectile collision
            elif proj["owner"] == "enemy":
                if (proj["pos"] - self.player["pos"]).length() < self.PLAYER_SIZE:
                    self.player["health"] -= proj["damage"]
                    collided = True
                    reward -= 0.02  # Penalty for getting hit
                    # sfx: player_hit.wav
                    self._create_hit_spark(proj["pos"], self.COLOR_PLAYER)

            if not collided:
                projectiles_to_keep.append(proj)

        self.projectiles = projectiles_to_keep
        return reward

    def _update_powerups(self):
        reward = 0
        for powerup in self.powerups[:]:
            if (
                self.player["pos"] - powerup["pos"]
            ).length() < self.PLAYER_SIZE + self.POWERUP_SIZE:
                reward += 0.5
                self.score += 5
                if powerup["type"] == "health":
                    self.player["health"] = min(
                        self.PLAYER_MAX_HEALTH, self.player["health"] + 25
                    )
                    # sfx: health_pickup.wav
                elif powerup["type"] == "speed":
                    self.player["speed_boost_timer"] = self.POWERUP_DURATION
                    # sfx: speed_pickup.wav
                elif powerup["type"] == "damage":
                    self.player["damage_boost_timer"] = self.POWERUP_DURATION
                    # sfx: damage_pickup.wav
                self.powerups.remove(powerup)
        return reward

    def _update_spawners(self):
        # Enemy respawning
        for i in range(len(self.dead_enemies_timers)):
            self.dead_enemies_timers[i] -= 1

        timers_ended = [t for t in self.dead_enemies_timers if t <= 0]
        self.dead_enemies_timers = [t for t in self.dead_enemies_timers if t > 0]
        for _ in timers_ended:
            self._spawn_enemy()

        # Powerup spawning
        if self.rng.random() < self.POWERUP_SPAWN_CHANCE and len(self.powerups) < 3:
            self._spawn_powerup()

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p.get("grow"):
                p["size"] += 0.5
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player["health"] <= 0:
            return True
        if self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT:
            return True
        return False

    def _spawn_enemy(self):
        side = self.rng.integers(0, 4)
        if side == 0:
            x, y = self.rng.integers(0, self.WIDTH), -self.ENEMY_SIZE
        elif side == 1:
            x, y = self.rng.integers(0, self.WIDTH), self.HEIGHT + self.ENEMY_SIZE
        elif side == 2:
            x, y = -self.ENEMY_SIZE, self.rng.integers(0, self.HEIGHT)
        else:
            x, y = self.WIDTH + self.ENEMY_SIZE, self.rng.integers(0, self.HEIGHT)

        self.enemies.append(
            {
                "pos": pygame.Vector2(x, y),
                "health": self.ENEMY_HEALTH,
                "speed": self.rng.uniform(0.8, 1.2),
                "shoot_cooldown": self.rng.integers(0, self.ENEMY_SHOOT_COOLDOWN),
            }
        )

    def _spawn_powerup(self):
        ptype = self.rng.choice(["health", "speed", "damage"])
        self.powerups.append(
            {
                "pos": pygame.Vector2(
                    self.rng.integers(50, self.WIDTH - 50),
                    self.rng.integers(50, self.HEIGHT - 50),
                ),
                "type": ptype,
            }
        )

    def _create_particle(self, pos, vel, life, color, size, grow=False):
        self.particles.append(
            {
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": life,
                "color": color,
                "size": size,
                "grow": grow,
            }
        )

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 360)
            speed = self.rng.uniform(1, 5)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            life = self.rng.integers(15, 30)
            size = self.rng.uniform(2, 5)
            self._create_particle(pos, vel, life, color, size)

    def _create_hit_spark(self, pos, color):
        for _ in range(5):
            angle = self.rng.uniform(0, 360)
            speed = self.rng.uniform(0.5, 2)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self._create_particle(pos, vel, 8, color, 2)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles (drawn first)
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], max(0, p["size"]))

        # Powerups
        for powerup in self.powerups:
            pos = (int(powerup["pos"].x), int(powerup["pos"].y))
            size = self.POWERUP_SIZE
            if powerup["type"] == "health":
                color = self.COLOR_HEALTH_POWERUP
            elif powerup["type"] == "speed":
                color = self.COLOR_SPEED_POWERUP
            else:
                color = self.COLOR_DAMAGE_POWERUP

            glow_size = size + 5 + int(math.sin(self.steps * 0.2) * 2)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY_GLOW
            )
            pygame.gfxdraw.aacircle(
                self.screen, pos[0], pos[1], self.ENEMY_SIZE, self.COLOR_ENEMY
            )
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], self.ENEMY_SIZE - 2, self.COLOR_ENEMY
            )

        # Player
        player_pos = (int(self.player["pos"].x), int(self.player["pos"].y))
        pygame.gfxdraw.filled_circle(
            self.screen,
            player_pos[0],
            player_pos[1],
            self.PLAYER_SIZE,
            self.COLOR_PLAYER_GLOW,
        )
        pygame.gfxdraw.aacircle(
            self.screen, player_pos[0], player_pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER
        )
        pygame.gfxdraw.filled_circle(
            self.screen,
            player_pos[0],
            player_pos[1],
            self.PLAYER_SIZE - 2,
            self.COLOR_PLAYER,
        )

        # Aiming reticle
        aim_vec = pygame.Vector2(self.PLAYER_SIZE + 5, 0).rotate(
            -self.player["aim_angle"]
        )
        reticle_pos = self.player["pos"] + aim_vec
        pygame.draw.circle(
            self.screen,
            self.COLOR_PLAYER_PROJ,
            (int(reticle_pos.x), int(reticle_pos.y)),
            2,
        )

        # Projectiles
        for proj in self.projectiles:
            color = (
                self.COLOR_PLAYER_PROJ
                if proj["owner"] == "player"
                else self.COLOR_ENEMY_PROJ
            )
            start_pos = proj["pos"] - proj["vel"] * 0.5
            end_pos = proj["pos"] + proj["vel"] * 0.5
            pygame.draw.aaline(self.screen, color, start_pos, end_pos, 3)

    def _render_ui(self):
        # Health bar
        health_ratio = np.clip(self.player["health"] / self.PLAYER_MAX_HEALTH, 0, 1)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(
            self.screen,
            self.COLOR_HEALTH_BAR,
            (10, 10, int(bar_width * health_ratio), 20),
        )

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Enemies defeated
        enemies_text = self.font_ui.render(
            f"DESTROYED: {self.enemies_defeated}/{self.TOTAL_ENEMIES_TO_DEFEAT}",
            True,
            self.COLOR_TEXT,
        )
        self.screen.blit(
            enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 35)
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "enemies_defeated": self.enemies_defeated,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It's a useful way to test and debug your environment
    env = GameEnv()
    obs, info = env.reset()

    running = True
    terminated = False
    truncated = False

    # Create a window to display the game
    pygame.display.set_caption("Robot Annihilation")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Player Input Handling ---
        keys = pygame.key.get_pressed()

        movement = 0  # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Frame Rate ---
        env.clock.tick(30)

    env.close()