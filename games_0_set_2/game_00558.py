import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Press space to attack. Survive the waves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in an isometric arena, collect coins, and get the highest score."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500  # Increased for longer waves
        self.FPS = 30

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_SHADOW = (10, 15, 20)
        self.COLOR_MONSTER = (255, 80, 80)
        self.COLOR_MONSTER_SHADOW = (10, 15, 20)
        self.COLOR_COIN = (255, 220, 50)
        self.COLOR_HEALTH_BG = (80, 80, 80)
        self.COLOR_HEALTH_PLAYER = (50, 255, 150)
        self.COLOR_HEALTH_MONSTER = (255, 80, 80)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_BG = (30, 40, 60, 180) # RGBA

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # --- Isometric Projection ---
        self.ISO_TILE_WIDTH = 48
        self.ISO_TILE_HEIGHT = 24
        self.GRID_SIZE_X = 14
        self.GRID_SIZE_Y = 14
        self.origin_x = self.WIDTH // 2
        self.origin_y = self.HEIGHT // 2 - 80

        # --- Game State (initialized in reset) ---
        self.player = {}
        self.monsters = []
        self.coins = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.last_hit_timer = 0
        self.combo_count = 0
        # self.np_random will be initialized in reset

        self.reset()
        self.validate_implementation()

    def _to_iso(self, cart_pos):
        iso_x = self.origin_x + (cart_pos.x - cart_pos.y) * self.ISO_TILE_WIDTH / 2
        iso_y = self.origin_y + (cart_pos.x + cart_pos.y) * self.ISO_TILE_HEIGHT / 2
        return pygame.math.Vector2(iso_x, iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Player State ---
        self.player = {
            "pos": pygame.math.Vector2(self.GRID_SIZE_X / 2, self.GRID_SIZE_Y / 2),
            "health": 100,
            "max_health": 100,
            "speed": 0.15,
            "facing": pygame.math.Vector2(0, 1),
            "attack_cooldown": 0,
            "invuln_timer": 0,
            "size": 0.4
        }

        # --- Entity Lists ---
        self.monsters = []
        self.coins = []
        self.particles = []

        # --- Game Counters ---
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.last_hit_timer = 0
        self.combo_count = 0

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        monster_speed = 0.04 + (self.wave - 1) * 0.005
        num_monsters = 15

        for _ in range(num_monsters):
            # Spawn monsters away from the center
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(self.GRID_SIZE_X * 0.3, self.GRID_SIZE_X * 0.45)
            pos_x = self.GRID_SIZE_X / 2 + math.cos(angle) * radius
            pos_y = self.GRID_SIZE_Y / 2 + math.sin(angle) * radius
            
            patrol_dir = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if patrol_dir.length() > 0:
                patrol_dir.normalize_ip()
            else:
                patrol_dir = pygame.math.Vector2(1,0)


            self.monsters.append({
                "pos": pygame.math.Vector2(pos_x, pos_y),
                "health": 20,
                "max_health": 20,
                "speed": monster_speed,
                "size": 0.35,
                "patrol_timer": 0,
                "patrol_dir": patrol_dir
            })
        # sfx: new_wave_chime.wav

    def step(self, action):
        reward = 0
        self.steps += 1
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement_action = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        # --- Update Cooldowns and Timers ---
        self.player["attack_cooldown"] = max(0, self.player["attack_cooldown"] - 1)
        self.player["invuln_timer"] = max(0, self.player["invuln_timer"] - 1)
        self.last_hit_timer = max(0, self.last_hit_timer - 1)
        if self.last_hit_timer == 0:
            self.combo_count = 0

        # --- Player Movement ---
        player_vel = pygame.math.Vector2(0, 0)
        if movement_action == 1: player_vel.y -= 1 # Up
        elif movement_action == 2: player_vel.y += 1 # Down
        elif movement_action == 3: player_vel.x -= 1 # Left
        elif movement_action == 4: player_vel.x += 1 # Right

        if player_vel.length() > 0:
            player_vel.normalize_ip()
            self.player["facing"] = player_vel.copy()

        old_player_pos = self.player["pos"].copy()
        self.player["pos"] += player_vel * self.player["speed"]

        # Clamp player to arena bounds
        self.player["pos"].x = np.clip(self.player["pos"].x, 0, self.GRID_SIZE_X - 1)
        self.player["pos"].y = np.clip(self.player["pos"].y, 0, self.GRID_SIZE_Y - 1)

        # --- Movement Reward ---
        if self.monsters:
            nearest_monster = min(self.monsters, key=lambda m: self.player["pos"].distance_to(m["pos"]))
            dist_before = old_player_pos.distance_to(nearest_monster["pos"])
            dist_after = self.player["pos"].distance_to(nearest_monster["pos"])
            if dist_after < dist_before:
                reward += 0.01 # Smaller reward for moving closer
            else:
                reward -= 0.002 # Smaller penalty

        # --- Player Attack ---
        if space_held and self.player["attack_cooldown"] == 0:
            self.player["attack_cooldown"] = 15 # Cooldown in frames
            attack_pos = self.player["pos"] + self.player["facing"] * 0.8
            attack_range = 1.2
            self._create_particles(self._to_iso(attack_pos), self.COLOR_PLAYER, 20, 10, 3) # Slash effect
            # sfx: player_attack_swoosh.wav
            
            hit_something = False
            for monster in self.monsters:
                if attack_pos.distance_to(monster["pos"]) < attack_range:
                    hit_something = True
                    monster["health"] -= 10
                    reward += 1.0 # Hit reward
                    reward += self.combo_count * 0.5 # Combo bonus
                    self.combo_count += 1
                    self.last_hit_timer = 45 # 1.5 seconds to continue combo
                    self._create_particles(self._to_iso(monster["pos"]), self.COLOR_MONSTER, 15, 5, 2)
                    # sfx: monster_hit.wav

        # --- Monster Logic ---
        monsters_to_remove = []
        for monster in self.monsters:
            # Movement
            dir_to_player = (self.player["pos"] - monster["pos"])
            if dir_to_player.length() > 0:
                dir_to_player.normalize_ip()
            monster["pos"] += dir_to_player * monster["speed"]
            
            # Monster-Player Collision
            if self.player["pos"].distance_to(monster["pos"]) < self.player["size"] + monster["size"] and self.player["invuln_timer"] == 0:
                self.player["health"] -= 5
                self.player["invuln_timer"] = 30 # 1s invulnerability
                self.combo_count = 0 # Reset combo on hit
                self._create_particles(self._to_iso(self.player["pos"]), (255, 0, 0), 30, 8, 4)
                # sfx: player_hurt.wav

            # Check for death
            if monster["health"] <= 0:
                monsters_to_remove.append(monster)
                reward += 5.0 # Kill reward
                self.score += 10
                self._spawn_coin(monster["pos"])
                self._create_particles(self._to_iso(monster["pos"]), self.COLOR_MONSTER, 50, 15, 5)
                # sfx: monster_death_explosion.wav

        self.monsters = [m for m in self.monsters if m not in monsters_to_remove]

        # --- Wave Completion ---
        if not self.monsters and not self.game_over:
            self.wave += 1
            reward += 50.0 # Wave clear bonus
            self._spawn_wave()
            # sfx: wave_complete_fanfare.wav

        # --- Coin Logic ---
        coins_to_remove = []
        for coin in self.coins:
            coin["lifetime"] -= 1
            if coin["lifetime"] <= 0:
                coins_to_remove.append(coin)
            if self.player["pos"].distance_to(coin["pos"]) < self.player["size"] + 0.5:
                coins_to_remove.append(coin)
                self.score += 10
                reward += 2.0 # Coin collection reward
                # sfx: coin_pickup.wav
        self.coins = [c for c in self.coins if c not in coins_to_remove]

        # --- Particle Logic ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] * 0.95)

        # --- Termination Conditions ---
        terminated = False
        if self.player["health"] <= 0:
            self.player["health"] = 0
            reward -= 50.0 # Death penalty
            terminated = True
            self.game_over = True
            # sfx: game_over_sound.wav
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_coin(self, pos):
        self.coins.append({
            "pos": pos.copy(),
            "lifetime": 300, # 10 seconds
            "anim_offset": self.np_random.uniform(0, math.pi * 2)
        })

    def _create_particles(self, pos, color, count, life, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(life // 2, life + 1),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })
            
    def _render_text(self, text, font, color, position, anchor="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, anchor, position)
        self.screen.blit(text_surface, text_rect)

    def _draw_health_bar(self, pos, size, health, max_health, bar_color):
        bg_rect = pygame.Rect(pos[0] - size[0] / 2, pos[1], size[0], size[1])
        health_ratio = np.clip(health / max_health, 0, 1)
        fg_rect = pygame.Rect(pos[0] - size[0] / 2, pos[1], size[0] * health_ratio, size[1])
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, bar_color, fg_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bg_rect, 1)

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render Grid ---
        for i in range(self.GRID_SIZE_X + 1):
            start = self._to_iso(pygame.math.Vector2(i, 0))
            end = self._to_iso(pygame.math.Vector2(i, self.GRID_SIZE_Y))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_SIZE_Y + 1):
            start = self._to_iso(pygame.math.Vector2(0, i))
            end = self._to_iso(pygame.math.Vector2(self.GRID_SIZE_X, i))
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # --- Sort and Render Entities ---
        entities = []
        entities.extend([("monster", m) for m in self.monsters])
        entities.extend([("coin", c) for c in self.coins])
        if not self.game_over:
            entities.append(("player", self.player))
        
        # Sort by cartesian y-coordinate for correct draw order
        entities.sort(key=lambda e: e[1]["pos"].y)

        for type, entity in entities:
            iso_pos = self._to_iso(entity["pos"])
            
            # Shadow
            shadow_pos = (int(iso_pos.x), int(iso_pos.y + 10))
            shadow_color = self.COLOR_PLAYER_SHADOW if type == "player" else self.COLOR_MONSTER_SHADOW
            if type != "coin":
                pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], 12, 5, shadow_color)

            if type == "player":
                # Glow effect
                if self.player["invuln_timer"] > 0 and self.steps % 4 < 2:
                    pass # flicker when invulnerable
                else:
                    for i in range(5, 0, -1):
                        alpha = 80 - i * 15
                        pygame.gfxdraw.filled_circle(self.screen, int(iso_pos.x), int(iso_pos.y), 10 + i, (*self.COLOR_PLAYER, alpha))
                    pygame.gfxdraw.filled_circle(self.screen, int(iso_pos.x), int(iso_pos.y), 10, self.COLOR_PLAYER)
                    pygame.gfxdraw.aacircle(self.screen, int(iso_pos.x), int(iso_pos.y), 10, self.COLOR_PLAYER)

            elif type == "monster":
                monster_size = 9
                monster_rect = pygame.Rect(iso_pos.x - monster_size, iso_pos.y - monster_size, monster_size*2, monster_size*2)
                pygame.draw.rect(self.screen, self.COLOR_MONSTER, monster_rect)
                self._draw_health_bar((iso_pos.x, iso_pos.y - 20), (30, 4), entity["health"], entity["max_health"], self.COLOR_HEALTH_MONSTER)

            elif type == "coin":
                coin_y_bob = math.sin(self.steps * 0.2 + entity["anim_offset"]) * 3
                pygame.gfxdraw.filled_circle(self.screen, int(iso_pos.x), int(iso_pos.y - coin_y_bob), 6, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, int(iso_pos.x), int(iso_pos.y - coin_y_bob), 6, self.COLOR_COIN)

        # --- Render Particles ---
        for p in self.particles:
            if p["size"] > 1:
                pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["size"]))

        # --- Render UI ---
        ui_panel = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        self._render_text(f"SCORE: {self.score}", self.font_large, self.COLOR_UI_TEXT, (10, 10))
        self._render_text(f"WAVE: {self.wave}", self.font_large, self.COLOR_UI_TEXT, (250, 10))
        if self.combo_count > 1:
            green_val = min(255, 150 + self.combo_count * 10)
            combo_color = (255, green_val, 0)
            self._render_text(f"COMBO x{self.combo_count}!", self.font_large, combo_color, (self.WIDTH / 2, 70), "midtop")
        
        # Player Health Bar
        self._draw_health_bar((self.WIDTH / 2, self.HEIGHT - 35), (200, 20), self.player["health"], self.player["max_health"], self.COLOR_HEALTH_PLAYER)

        if self.game_over:
            self._render_text("GAME OVER", pygame.font.Font(None, 80), self.COLOR_MONSTER, (self.WIDTH / 2, self.HEIGHT / 2), "center")

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player["health"],
            "monsters_left": len(self.monsters),
            "combo": self.combo_count
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        # Check that step with game over returns correct truncated value
        self.steps = self.MAX_STEPS
        obs, reward, term, trunc, info = self.step(test_action)
        assert trunc, "Truncated should be True when max_steps is reached"

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # To run, ensure you have a display environment or comment out the display lines.
    # For example, by unsetting the dummy video driver:
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Arena")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_bar = 0
        shift_key = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_bar = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_key = 1
            
        action = [movement, space_bar, shift_key]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()