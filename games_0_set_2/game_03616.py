import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot in your last moved direction. Survive the horde!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde for 60 seconds in a top-down arena. "
        "Shoot zombies, collect health packs, and stay alive!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 32
    GRID_WIDTH = 18
    GRID_HEIGHT = 10
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    PLAYER_MAX_HEALTH = 100
    PLAYER_MOVE_COOLDOWN = 4
    PLAYER_SHOOT_COOLDOWN = 8
    PLAYER_IFRAMES = 15  # Invincibility frames after taking damage

    ZOMBIE_MOVE_COOLDOWN = 15
    ZOMBIE_DAMAGE = 10
    INITIAL_ZOMBIE_COUNT = 20
    ZOMBIE_SPAWN_CAP_INCREASE_INTERVAL = 15 * FPS

    HEALTH_PACK_VALUE = 25
    HEALTH_PACK_SPAWN_INTERVAL = 10 * FPS
    HEALTH_PACK_MAX = 2

    PROJECTILE_SPEED = 18

    # Colors
    COLOR_BG = (10, 5, 10)
    COLOR_GRID = (30, 20, 30)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_FACING = (150, 255, 150)
    COLOR_PLAYER_HIT = (255, 100, 100)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_HEALTH_PACK = (50, 150, 255)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_BLOOD = (120, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (200, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = None
        self.player_health = None
        self.player_facing_direction = None
        self.player_move_cooldown_timer = None
        self.player_shoot_cooldown_timer = None
        self.player_iframe_timer = None

        self.zombies = None
        self.zombie_move_cooldown_timer = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_cap = None

        self.projectiles = None
        self.health_packs = None
        self.health_pack_spawn_timer = None
        self.particles = None

        self.steps = None
        self.score = None
        self.kill_count = None
        self.game_over_message = None

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return pygame.Vector2(x, y)

    def _get_random_edge_spawn(self):
        side = self.np_random.integers(4)
        if side == 0:  # Top
            return (self.np_random.integers(self.GRID_WIDTH), -1)
        elif side == 1:  # Bottom
            return (self.np_random.integers(self.GRID_WIDTH), self.GRID_HEIGHT)
        elif side == 2:  # Left
            return (-1, self.np_random.integers(self.GRID_HEIGHT))
        else:  # Right
            return (self.GRID_WIDTH, self.np_random.integers(self.GRID_HEIGHT))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.kill_count = 0
        self.game_over_message = ""

        self.player_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = pygame.Vector2(0, -1)  # Default up
        self.player_move_cooldown_timer = 0
        self.player_shoot_cooldown_timer = 0
        self.player_iframe_timer = 0

        self.zombies = []
        self.zombie_move_cooldown_timer = 0
        self.zombie_spawn_cap = self.INITIAL_ZOMBIE_COUNT
        self.zombie_spawn_timer = 0

        self.projectiles = []
        self.health_packs = []
        self.health_pack_spawn_timer = self.HEALTH_PACK_SPAWN_INTERVAL // 2
        self.particles = []

        # Initial spawns
        occupied_cells = {tuple(self.player_pos)}
        for _ in range(self.INITIAL_ZOMBIE_COUNT):
            pos = self._get_random_edge_spawn()
            self.zombies.append(pygame.Vector2(pos))

        while len(self.health_packs) < 1:
            pos = (self.np_random.integers(self.GRID_WIDTH), self.np_random.integers(self.GRID_HEIGHT))
            if pos not in occupied_cells:
                self.health_packs.append(pygame.Vector2(pos))
                occupied_cells.add(pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if self.player_health <= 0:
            self.game_over_message = "YOU DIED"
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over_message = "YOU SURVIVED!"
            reward += 100.0
            terminated = True

        if not terminated:
            reward += 0.1  # Survival reward
            self._handle_input(action)
            self._update_player()
            self._update_zombies()
            self._update_projectiles()
            self._update_health_packs()

            collision_rewards = self._handle_collisions()
            reward += collision_rewards

        self._update_particles()

        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action

        # Movement
        if self.player_move_cooldown_timer == 0:
            move_dir = pygame.Vector2(0, 0)
            if movement == 1: move_dir.y = -1  # Up
            elif movement == 2: move_dir.y = 1  # Down
            elif movement == 3: move_dir.x = -1  # Left
            elif movement == 4: move_dir.x = 1  # Right

            if move_dir.length() > 0:
                new_pos = self.player_pos + move_dir
                if 0 <= new_pos.x < self.GRID_WIDTH and 0 <= new_pos.y < self.GRID_HEIGHT:
                    self.player_pos = new_pos
                    self.player_move_cooldown_timer = self.PLAYER_MOVE_COOLDOWN
                self.player_facing_direction = pygame.Vector2(move_dir)

        # Shooting
        if space_held and self.player_shoot_cooldown_timer == 0:
            # sfx: player_shoot.wav
            start_pos = self._grid_to_pixel(self.player_pos)
            self.projectiles.append({
                "pos": pygame.Vector2(start_pos),
                "dir": pygame.Vector2(self.player_facing_direction),
            })
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
            # Recoil particle effect
            for _ in range(5):
                self._create_particle(start_pos, self.COLOR_PROJECTILE, 1, 4, 5, -self.player_facing_direction)

    def _update_player(self):
        if self.player_move_cooldown_timer > 0: self.player_move_cooldown_timer -= 1
        if self.player_shoot_cooldown_timer > 0: self.player_shoot_cooldown_timer -= 1
        if self.player_iframe_timer > 0: self.player_iframe_timer -= 1

    def _update_zombies(self):
        # Spawning
        self.zombie_spawn_timer += 1
        if len(self.zombies) < self.zombie_spawn_cap:
            spawn_interval = max(5, self.FPS - (self.steps // (self.FPS * 5)))
            if self.zombie_spawn_timer >= spawn_interval:
                self.zombies.append(pygame.Vector2(self._get_random_edge_spawn()))
                self.zombie_spawn_timer = 0

        if self.steps > 0 and self.steps % self.ZOMBIE_SPAWN_CAP_INCREASE_INTERVAL == 0:
            self.zombie_spawn_cap += 1

        # Movement
        self.zombie_move_cooldown_timer -= 1
        if self.zombie_move_cooldown_timer <= 0:
            self.zombie_move_cooldown_timer = self.ZOMBIE_MOVE_COOLDOWN
            for z in self.zombies:
                dx, dy = self.player_pos.x - z.x, self.player_pos.y - z.y
                if abs(dx) > abs(dy):
                    z.x += np.sign(dx)
                elif dy != 0:
                    z.y += np.sign(dy)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["dir"] * self.PROJECTILE_SPEED
            if not self.screen.get_rect().collidepoint(p["pos"]):
                self.projectiles.remove(p)

    def _update_health_packs(self):
        self.health_pack_spawn_timer -= 1
        if self.health_pack_spawn_timer <= 0 and len(self.health_packs) < self.HEALTH_PACK_MAX:
            self.health_pack_spawn_timer = self.HEALTH_PACK_SPAWN_INTERVAL
            occupied_cells = {tuple(z) for z in self.zombies}
            occupied_cells.add(tuple(self.player_pos))
            for hp in self.health_packs: occupied_cells.add(tuple(hp))

            attempts = 0
            while attempts < 50:
                pos = pygame.Vector2(self.np_random.integers(self.GRID_WIDTH), self.np_random.integers(self.GRID_HEIGHT))
                if tuple(pos) not in occupied_cells:
                    self.health_packs.append(pos)
                    # sfx: health_pack_spawn.wav
                    break
                attempts += 1

    def _create_particle(self, pos, color, min_speed, max_speed, lifespan, direction=None):
        if direction is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_dir = pygame.Vector2(math.cos(angle), math.sin(angle))
        else:
            angle = math.atan2(direction.y, direction.x) + self.np_random.uniform(-0.5, 0.5)
            vel_dir = pygame.Vector2(math.cos(angle), math.sin(angle))

        speed = self.np_random.uniform(min_speed, max_speed)
        velocity = vel_dir * speed
        self.particles.append({
            "pos": pygame.Vector2(pos),
            "vel": velocity,
            "lifespan": lifespan,
            "max_lifespan": lifespan,
            "color": color
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9  # Damping
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0

        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            for z in self.zombies[:]:
                if self._grid_to_pixel(z).distance_to(p["pos"]) < self.CELL_SIZE // 2:
                    # sfx: zombie_die.wav
                    self.zombies.remove(z)
                    if p in self.projectiles: self.projectiles.remove(p)
                    reward += 1.0
                    self.kill_count += 1
                    for _ in range(30):  # Blood splatter
                        self._create_particle(self._grid_to_pixel(z), self.COLOR_BLOOD, 2, 6, 20)
                    break

        # Player vs Zombies
        if self.player_iframe_timer == 0:
            player_pixel_pos = self._grid_to_pixel(self.player_pos)
            for z in self.zombies:
                if player_pixel_pos.distance_to(self._grid_to_pixel(z)) < self.CELL_SIZE * 0.8:
                    # sfx: player_hit.wav
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.player_iframe_timer = self.PLAYER_IFRAMES
                    for _ in range(20):  # Player hit sparks
                        self._create_particle(player_pixel_pos, self.COLOR_PLAYER_HIT, 2, 5, 15)
                    break

        # Player vs Health Packs
        for hp in self.health_packs[:]:
            if self.player_pos == hp:
                # sfx: collect_health.wav
                self.player_health = min(self.PLAYER_MAX_HEALTH, self.player_health + self.HEALTH_PACK_VALUE)
                self.health_packs.remove(hp)
                reward += 5.0
                break

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"]
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 3, (*color, alpha))

        # Health Packs
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        hp_size = int(self.CELL_SIZE * 0.3 + pulse * 3)
        for hp in self.health_packs:
            pos = self._grid_to_pixel(hp)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), hp_size, self.COLOR_HEALTH_PACK)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), hp_size, self.COLOR_HEALTH_PACK)

        # Zombies
        zombie_size = self.CELL_SIZE // 2 - 2
        for z in self.zombies:
            pos = self._grid_to_pixel(z)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), zombie_size, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), zombie_size, self.COLOR_ZOMBIE)

        # Player
        player_color = self.COLOR_PLAYER
        if self.player_iframe_timer > 0 and self.steps % 4 < 2:
            player_color = self.COLOR_PLAYER_HIT

        player_pixel_pos = self._grid_to_pixel(self.player_pos)
        player_size = self.CELL_SIZE // 2 - 2
        pygame.gfxdraw.filled_circle(self.screen, int(player_pixel_pos.x), int(player_pixel_pos.y), player_size,
                                      player_color)
        pygame.gfxdraw.aacircle(self.screen, int(player_pixel_pos.x), int(player_pixel_pos.y), player_size,
                                player_color)

        # Player facing indicator
        facing_pos = player_pixel_pos + self.player_facing_direction * (player_size * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, int(facing_pos.x), int(facing_pos.y), 4, self.COLOR_PLAYER_FACING)

        # Projectiles
        for p in self.projectiles:
            end_pos = p["pos"] - p["dir"] * 10
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (p["pos"].x, p["pos"].y), (end_pos.x, end_pos.y), 3)

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_bar_height = 15
        health_bar_x = (self.SCREEN_WIDTH - health_bar_width) // 2
        health_bar_y = self.SCREEN_HEIGHT - health_bar_height - 10

        current_health_width = (self.player_health / self.PLAYER_MAX_HEALTH) * health_bar_width

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG,
                         (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG,
                             (health_bar_x, health_bar_y, current_health_width, health_bar_height))

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = f"TIME: {time_left:.1f}"
        text_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))

        # Kill Count
        kill_text = f"KILLS: {self.kill_count}"
        text_surf = self.font_small.render(kill_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Game Over Message
        if self.game_over_message:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text_surf = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "kills": self.kill_count,
            "health": self.player_health,
            "time_left": max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # The validation code that was in __init__ is now here
    def validate_implementation(env_instance):
        print("Validating implementation...")
        # Test action space
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        obs, info = env_instance.reset()
        assert obs.shape == (env_instance.SCREEN_HEIGHT, env_instance.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8

        # Test reset
        assert isinstance(info, dict)

        # Test step
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (env_instance.SCREEN_HEIGHT, env_instance.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

    env = GameEnv()
    validate_implementation(env)

    # For interactive play, you need a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Kills: {info['kills']}")
            pygame.time.wait(3000)  # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()