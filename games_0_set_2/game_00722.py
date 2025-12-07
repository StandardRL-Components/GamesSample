
# Generated: 2025-08-27T14:34:17.166926
# Source Brief: brief_00722.md
# Brief Index: 722

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to attack in your last moved direction. Hold Shift to dodge."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a fast-paced isometric arena. Dodge attacks and time your strikes to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals & Constants
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_ARENA = (52, 73, 94)
        self.COLOR_PLAYER = (46, 204, 113)
        self.COLOR_PLAYER_DODGE = (120, 230, 170)
        self.COLOR_ENEMY_W1 = (231, 76, 60)
        self.COLOR_ENEMY_W2 = (230, 126, 34)
        self.COLOR_ENEMY_W3 = (155, 89, 182)
        self.COLOR_ATTACK = (241, 196, 15)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_DAMAGE = (255, 255, 255)
        self.COLOR_WIN = (46, 204, 113)
        self.COLOR_LOSE = (231, 76, 60)
        
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_ATTACK_COOLDOWN = 10 # frames
        self.PLAYER_DODGE_COOLDOWN = 60 # frames
        self.PLAYER_DODGE_DURATION = 15 # frames
        self.PROJECTILE_SPEED = 8

        self.ARENA_BOUNDS = pygame.Rect(100, 50, 440, 300)

        # Initialize state variables
        self.np_random = None
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.damage_texts = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.player = {
            "pos": pygame.Vector2(self.screen_width / 2, self.screen_height / 2),
            "health": self.PLAYER_HEALTH_MAX,
            "last_move_dir": pygame.Vector2(0, -1),
            "attack_cooldown": 0,
            "dodge_cooldown": 0,
            "dodge_timer": 0,
            "is_dodging": False
        }
        
        self.wave_num = 0
        self.wave_transition_timer = 30 # Start with a short delay for the first wave
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.damage_texts = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self.wave_num += 1
                reward += self._spawn_wave()
        else:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_pressed = action[1] == 1
            shift_pressed = action[2] == 1
            
            # Update player
            self._update_player(movement, space_pressed, shift_pressed)
            
            # Update enemies
            self._update_enemies()

        # Update projectiles and effects
        self._update_projectiles()
        self._update_particles()
        self._update_damage_texts()

        # Collision detection
        reward += self._handle_collisions()
        
        # Check for wave completion
        if not self.enemies and self.wave_transition_timer == 0 and self.wave_num > 0 and self.wave_num <= 3:
            if self.wave_num == 3: # Final wave cleared
                self.game_over = True
                self.win_condition = True
                reward += 50
            else:
                self.wave_transition_timer = 90 # 3 second pause
                reward += 5

        # Check termination conditions
        terminated = False
        if self.player["health"] <= 0:
            self.player["health"] = 0
            self.game_over = True
            self.win_condition = False
            reward -= 50 # Large penalty for dying
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_condition = self.player["health"] > 0 and self.wave_num == 3 and not self.enemies
            terminated = True
        elif self.game_over and self.win_condition:
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_wave(self):
        wave_configs = {
            1: {"count": 3, "health": 50, "attack_rate": 60, "color": self.COLOR_ENEMY_W1},
            2: {"count": 4, "health": 75, "attack_rate": 45, "color": self.COLOR_ENEMY_W2},
            3: {"count": 5, "health": 100, "attack_rate": 30, "color": self.COLOR_ENEMY_W3},
        }
        if self.wave_num not in wave_configs:
            return 0

        config = wave_configs[self.wave_num]
        for _ in range(config["count"]):
            self.enemies.append({
                "pos": pygame.Vector2(
                    self.np_random.integers(self.ARENA_BOUNDS.left, self.ARENA_BOUNDS.right),
                    self.np_random.integers(self.ARENA_BOUNDS.top, self.ARENA_BOUNDS.top + 100)
                ),
                "health": config["health"],
                "max_health": config["health"],
                "attack_cooldown": self.np_random.integers(0, config["attack_rate"]),
                "attack_rate": config["attack_rate"],
                "color": config["color"]
            })
        return 5 # Wave start bonus

    def _update_player(self, movement, space_pressed, shift_pressed):
        # Cooldowns
        if self.player["attack_cooldown"] > 0: self.player["attack_cooldown"] -= 1
        if self.player["dodge_cooldown"] > 0: self.player["dodge_cooldown"] -= 1

        # Dodge state
        if self.player["is_dodging"]:
            self.player["dodge_timer"] -= 1
            if self.player["dodge_timer"] <= 0:
                self.player["is_dodging"] = False
        
        # Actions
        if shift_pressed and self.player["dodge_cooldown"] == 0 and not self.player["is_dodging"]:
            self.player["is_dodging"] = True
            self.player["dodge_timer"] = self.PLAYER_DODGE_DURATION
            self.player["dodge_cooldown"] = self.PLAYER_DODGE_COOLDOWN
            # sfx: player_dodge.wav
            dodge_dir = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
            self.player["pos"] += dodge_dir * self.PLAYER_SPEED * 3
            for _ in range(10): self._create_particle(self.player["pos"], self.COLOR_PLAYER_DODGE, 2, 10)


        if space_pressed and self.player["attack_cooldown"] == 0 and not self.player["is_dodging"]:
            self.player["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN
            # sfx: player_attack.wav
            proj_pos = self.player["pos"] + self.player["last_move_dir"] * 20
            self.player_projectiles.append({"pos": proj_pos, "dir": self.player["last_move_dir"]})
            self._create_particle(proj_pos, self.COLOR_ATTACK, 5, 5, count=5)


        # Movement
        move_dir = pygame.Vector2(0, 0)
        if not self.player["is_dodging"]:
            if movement == 1: move_dir.y = -1 # Up
            elif movement == 2: move_dir.y = 1 # Down
            elif movement == 3: move_dir.x = -1 # Left
            elif movement == 4: move_dir.x = 1 # Right
            
            if move_dir.length() > 0:
                move_dir.normalize_ip()
                self.player["last_move_dir"] = move_dir.copy()
                self.player["pos"] += move_dir * self.PLAYER_SPEED
        
        # Clamp player position to arena
        self.player["pos"].x = np.clip(self.player["pos"].x, self.ARENA_BOUNDS.left, self.ARENA_BOUNDS.right)
        self.player["pos"].y = np.clip(self.player["pos"].y, self.ARENA_BOUNDS.top, self.ARENA_BOUNDS.bottom)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["attack_cooldown"] -= 1
            if enemy["attack_cooldown"] <= 0:
                enemy["attack_cooldown"] = enemy["attack_rate"]
                dir_to_player = (self.player["pos"] - enemy["pos"]).normalize()
                # sfx: enemy_attack.wav
                self.enemy_projectiles.append({"pos": enemy["pos"].copy(), "dir": dir_to_player})

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if self.ARENA_BOUNDS.collidepoint(p["pos"])]
        for p in self.player_projectiles:
            p["pos"] += p["dir"] * self.PROJECTILE_SPEED

        self.enemy_projectiles = [p for p in self.enemy_projectiles if self.screen.get_rect().inflate(50, 50).collidepoint(p["pos"])]
        for p in self.enemy_projectiles:
            p["pos"] += p["dir"] * self.PROJECTILE_SPEED

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj["pos"].distance_to(enemy["pos"]) < 15:
                    damage = 25
                    enemy["health"] -= damage
                    reward += 0.1
                    self._create_damage_text(enemy["pos"], str(damage))
                    self._create_particle(proj["pos"], self.COLOR_ATTACK, 4, 15, count=10)
                    # sfx: hit_confirm.wav
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    if enemy["health"] <= 0:
                        reward += 1
                        # sfx: enemy_die.wav
                        self._create_particle(enemy["pos"], enemy["color"], 8, 20, count=30)
                        self.enemies.remove(enemy)
                    break
        
        # Enemy projectiles vs player
        if not self.player["is_dodging"]:
            player_rect = pygame.Rect(self.player["pos"].x-7, self.player["pos"].y-7, 14, 14)
            for proj in self.enemy_projectiles[:]:
                if player_rect.collidepoint(proj["pos"]):
                    self.player["health"] -= 10
                    reward -= 0.1
                    # sfx: player_hit.wav
                    self._create_particle(self.player["pos"], self.COLOR_PLAYER, 4, 15, count=10)
                    if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, self.ARENA_BOUNDS, border_radius=10)
        
        # Sort entities by y-position for correct draw order
        entities = [{"type": "player", **self.player}] + [{"type": "enemy", **e} for e in self.enemies]
        entities.sort(key=lambda e: e["pos"].y)

        # Draw shadows first
        for entity in entities:
            shadow_pos = (int(entity["pos"].x), int(entity["pos"].y) + 12)
            shadow_surf = pygame.Surface((30, 15), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, shadow_surf.get_rect())
            self.screen.blit(shadow_surf, shadow_surf.get_rect(center=shadow_pos))

        # Draw entities
        for entity in entities:
            if entity["type"] == "player":
                self._draw_character(self.player["pos"], self.COLOR_PLAYER, self.player["is_dodging"])
            elif entity["type"] == "enemy":
                self._draw_character(entity["pos"], entity["color"], False)
                # Enemy health bar
                bar_w = 30
                bar_h = 5
                health_pct = max(0, entity["health"] / entity["max_health"])
                pygame.draw.rect(self.screen, (50,50,50), (entity["pos"].x - bar_w/2, entity["pos"].y - 25, bar_w, bar_h))
                pygame.draw.rect(self.screen, entity["color"], (entity["pos"].x - bar_w/2, entity["pos"].y - 25, bar_w * health_pct, bar_h))

        # Draw projectiles
        for p in self.player_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ATTACK, (int(p["pos"].x), int(p["pos"].y)), 5)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), 5, self.COLOR_ATTACK)
        for p in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_W1, (int(p["pos"].x), int(p["pos"].y)), 4)
        
        self._render_particles()
        self._render_damage_texts()

    def _draw_character(self, pos, color, is_dodging):
        size = 15
        if is_dodging:
            alpha_color = (*color, 100)
            pygame.draw.rect(self.screen, alpha_color, (pos.x - size/2, pos.y - size/2, size, size))
        else:
            pygame.draw.rect(self.screen, color, (pos.x - size/2, pos.y - size/2, size, size))
        
    def _render_ui(self):
        # Player Health
        health_bar_w = 200
        health_bar_h = 20
        health_pct = max(0, self.player["health"] / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, health_bar_w, health_bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 10, health_bar_w * health_pct, health_bar_h))
        health_text = self.font_small.render(f'HP: {int(self.player["health"])}/{self.PLAYER_HEALTH_MAX}', True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_medium.render(f'Score: {int(self.score)}', True, self.COLOR_TEXT)
        self.screen.blit(score_text, score_text.get_rect(centerx=self.screen_width/2, y=10))

        # Wave
        wave_str = f'Wave: {self.wave_num}/3' if self.wave_num > 0 else 'Get Ready!'
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, wave_text.get_rect(right=self.screen_width - 10, y=10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            if self.win_condition:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.screen_width/2, self.screen_height/2)))
    
    def _create_particle(self, pos, color, size, lifespan, count=1):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "color": color,
                "size": size,
                "lifespan": lifespan
            })

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 10))
            alpha = max(0, min(255, alpha))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, temp_surf.get_rect(center=p["pos"]))
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _create_damage_text(self, pos, text):
        self.damage_texts.append({
            "pos": pos.copy(),
            "text": text,
            "lifespan": 30
        })

    def _render_damage_texts(self):
        for dt in self.damage_texts:
            alpha = int(255 * (dt["lifespan"] / 30))
            text_surf = self.font_small.render(dt["text"], True, self.COLOR_DAMAGE)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_surf.get_rect(center=dt["pos"]))
    
    def _update_damage_texts(self):
        for dt in self.damage_texts:
            dt["pos"].y -= 1
            dt["lifespan"] -= 1
        self.damage_texts = [dt for dt in self.damage_texts if dt["lifespan"] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "wave": self.wave_num,
            "enemies_left": len(self.enemies)
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert self.wave_num == 0 # Should start before wave 1
        assert self.wave_transition_timer > 0 # Should be in transition
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test termination assertions
        self.player["health"] = 0
        _, _, term, _, _ = self.step(self.action_space.sample())
        assert term == True
        self.reset()
        
        self.steps = self.MAX_STEPS
        _, _, term, _, _ = self.step(self.action_space.sample())
        assert term == True
        self.reset()
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    env = GameEnv()
    obs, info = env.reset()
    
    terminated = False
    running = True
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Isometric Arena Fighter")
    
    action = env.action_space.sample() # Start with a default action
    action[0] = 0 # No movement initially
    action[1] = 0 # No space
    action[2] = 0 # No shift

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Reset action each frame
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        # ----------------------
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment returns an array, but for display we use the surface directly
        # The _get_observation call inside step already drew everything to env.screen
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        env.clock.tick(30) # Run at 30 FPS

    env.close()