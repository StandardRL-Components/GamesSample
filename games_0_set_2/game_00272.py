
# Generated: 2025-08-27T13:08:27.099155
# Source Brief: brief_00272.md
# Brief Index: 272

        
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
        "Controls: Arrow keys to move your tank. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Command a tiny tank in a top-down arena. Strategically maneuver and fire to destroy the enemy base while defending your own."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500 # 50 seconds at 30fps

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_PLAYER_BASE = (40, 100, 200)
    COLOR_ENEMY_BASE = (200, 60, 60)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_EXPLOSION = (255, 200, 0)
    COLOR_WHITE = (220, 220, 220)
    COLOR_HEALTH_BG = (80, 0, 0)
    COLOR_HEALTH_FG = (0, 180, 0)
    COLOR_UI_TEXT = (240, 240, 240)

    # Game Parameters
    TANK_RADIUS = 12
    TANK_SPEED = 4
    TANK_HEALTH = 100
    TANK_FIRE_COOLDOWN = 10 # frames
    BASE_SIZE = 40
    BASE_HEALTH = 200
    PROJECTILE_SIZE = 4
    PROJECTILE_SPEED = 12
    PROJECTILE_DAMAGE = 10
    AI_SIGHT_RANGE = 300
    AI_FIRE_COOLDOWN = 35 # Slower than player

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
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 14)
        self.font_large = pygame.font.SysFont("sans", 24)
        
        # Initialize state variables
        self.player_tank = {}
        self.enemy_tank = {}
        self.player_base = {}
        self.enemy_base = {}
        self.projectiles = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "VICTORY" or "DEFEAT"
        
        # Initialize state
        self.reset()

        # Validate implementation
        # self.validate_implementation() # Optional: uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        self.player_tank = {
            "pos": np.array([100.0, self.HEIGHT / 2.0]),
            "health": self.TANK_HEALTH,
            "dir": np.array([1.0, 0.0]),
            "cooldown": 0
        }
        self.enemy_tank = {
            "pos": np.array([self.WIDTH - 100.0, self.HEIGHT / 2.0]),
            "health": self.TANK_HEALTH,
            "dir": np.array([-1.0, 0.0]),
            "cooldown": 0,
            "patrol_dir": 1 # 1 for down, -1 for up
        }
        
        self.player_base = {
            "pos": np.array([self.BASE_SIZE / 2, self.HEIGHT / 2]),
            "health": self.BASE_HEALTH
        }
        self.enemy_base = {
            "pos": np.array([self.WIDTH - self.BASE_SIZE / 2, self.HEIGHT / 2]),
            "health": self.BASE_HEALTH
        }

        self.projectiles = []
        self.explosions = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty for each step to encourage efficiency
        
        # --- Update Cooldowns ---
        if self.player_tank["cooldown"] > 0:
            self.player_tank["cooldown"] -= 1
        if self.enemy_tank["cooldown"] > 0:
            self.enemy_tank["cooldown"] -= 1

        # --- Handle Player Action ---
        movement = action[0]
        space_held = action[1] == 1
        
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0  # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0  # Right

        if np.linalg.norm(move_vec) > 0:
            self.player_tank["dir"] = move_vec
            self.player_tank["pos"] += move_vec * self.TANK_SPEED
        
        # Clamp player position
        self.player_tank["pos"][0] = np.clip(self.player_tank["pos"][0], self.TANK_RADIUS, self.WIDTH - self.TANK_RADIUS)
        self.player_tank["pos"][1] = np.clip(self.player_tank["pos"][1], self.TANK_RADIUS, self.HEIGHT - self.TANK_RADIUS)

        # Player firing
        if space_held and self.player_tank["cooldown"] == 0 and self.player_tank["health"] > 0:
            self._fire_projectile(self.player_tank, "player")
            self.player_tank["cooldown"] = self.TANK_FIRE_COOLDOWN
            # sfx: player_shoot

        # --- Update Enemy AI ---
        reward += self._update_enemy_ai()

        # --- Update Projectiles ---
        new_projectiles = []
        for p in self.projectiles:
            p["pos"] += p["dir"] * self.PROJECTILE_SPEED
            
            # Check boundaries
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                continue # Projectile is out of bounds, discard

            # Check collisions
            collided = False
            if p["owner"] == "player":
                # Vs Enemy Tank
                if self.enemy_tank["health"] > 0 and np.linalg.norm(p["pos"] - self.enemy_tank["pos"]) < self.TANK_RADIUS + self.PROJECTILE_SIZE:
                    self.enemy_tank["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1
                    self._create_explosion(p["pos"])
                    collided = True
                    if self.enemy_tank["health"] <= 0:
                        reward += 5
                        self.enemy_tank["health"] = 0
                        self._create_explosion(self.enemy_tank["pos"], large=True)
                        # sfx: big_explosion
                # Vs Enemy Base
                elif self._check_base_collision(p["pos"], self.enemy_base):
                    self.enemy_base["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.2
                    self._create_explosion(p["pos"])
                    collided = True
                    # sfx: base_hit
            
            elif p["owner"] == "enemy":
                 # Vs Player Tank
                if self.player_tank["health"] > 0 and np.linalg.norm(p["pos"] - self.player_tank["pos"]) < self.TANK_RADIUS + self.PROJECTILE_SIZE:
                    self.player_tank["health"] -= self.PROJECTILE_DAMAGE
                    reward -= 0.1 # Negative reward for getting hit
                    self._create_explosion(p["pos"])
                    collided = True
                    if self.player_tank["health"] <= 0:
                        self.player_tank["health"] = 0
                        self._create_explosion(self.player_tank["pos"], large=True)
                        # sfx: big_explosion
                # Vs Player Base
                elif self._check_base_collision(p["pos"], self.player_base):
                    self.player_base["health"] -= self.PROJECTILE_DAMAGE
                    reward -= 0.2
                    self._create_explosion(p["pos"])
                    collided = True
                    # sfx: base_hit

            if not collided:
                new_projectiles.append(p)

        self.projectiles = new_projectiles
        
        # --- Update Explosions ---
        self.explosions = [e for e in self.explosions if e["life"] > 0]
        for e in self.explosions:
            e["life"] -= 1

        # --- Update Game State ---
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if self.enemy_base["health"] <= 0:
                reward += 100
                self.game_outcome = "VICTORY"
            elif self.player_base["health"] <= 0:
                reward -= 100
                self.game_outcome = "DEFEAT"
            self.score += reward # Add terminal reward to final score
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_enemy_ai(self):
        if self.enemy_tank["health"] <= 0:
            return 0

        # --- Movement AI (Patrol) ---
        patrol_top = 80
        patrol_bottom = self.HEIGHT - 80
        
        self.enemy_tank["pos"][1] += self.enemy_tank["patrol_dir"] * (self.TANK_SPEED * 0.75) # Move slightly slower
        if self.enemy_tank["pos"][1] >= patrol_bottom:
            self.enemy_tank["patrol_dir"] = -1
        elif self.enemy_tank["pos"][1] <= patrol_top:
            self.enemy_tank["patrol_dir"] = 1

        # --- Firing AI ---
        dist_to_player = np.linalg.norm(self.player_tank["pos"] - self.enemy_tank["pos"])
        
        if dist_to_player < self.AI_SIGHT_RANGE and self.enemy_tank["cooldown"] == 0:
            # Aim at player
            direction_to_player = (self.player_tank["pos"] - self.enemy_tank["pos"])
            norm = np.linalg.norm(direction_to_player)
            if norm > 0:
                self.enemy_tank["dir"] = direction_to_player / norm
            
            self._fire_projectile(self.enemy_tank, "enemy")
            self.enemy_tank["cooldown"] = self.AI_FIRE_COOLDOWN
            # sfx: enemy_shoot
        
        return 0 # AI actions don't generate immediate reward

    def _check_base_collision(self, proj_pos, base):
        base_rect = pygame.Rect(
            base["pos"][0] - self.BASE_SIZE / 2,
            base["pos"][1] - self.BASE_SIZE / 2,
            self.BASE_SIZE, self.BASE_SIZE
        )
        return base_rect.collidepoint(proj_pos[0], proj_pos[1])

    def _fire_projectile(self, owner_tank, owner_type):
        start_pos = owner_tank["pos"] + owner_tank["dir"] * (self.TANK_RADIUS + 5)
        self.projectiles.append({
            "pos": start_pos,
            "dir": owner_tank["dir"],
            "owner": owner_type
        })
    
    def _create_explosion(self, pos, large=False):
        self.explosions.append({
            "pos": pos,
            "life": 10 if not large else 20,
            "max_life": 10 if not large else 20,
            "max_radius": 15 if not large else 40
        })
        # sfx: small_explosion

    def _check_termination(self):
        if self.player_base["health"] <= 0 or self.enemy_base["health"] <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Bases
        self._render_base(self.player_base, self.COLOR_PLAYER_BASE, "PLAYER BASE")
        self._render_base(self.enemy_base, self.COLOR_ENEMY_BASE, "ENEMY BASE")

        # Render Projectiles
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0] - 2, pos[1] - 2, 4, 4))
        
        # Render Tanks
        if self.player_tank["health"] > 0:
            self._render_tank(self.player_tank, self.COLOR_PLAYER)
        if self.enemy_tank["health"] > 0:
            self._render_tank(self.enemy_tank, self.COLOR_ENEMY)
        
        # Render Explosions
        for e in self.explosions:
            progress = e["life"] / e["max_life"]
            radius = int(e["max_radius"] * (1 - progress))
            alpha = int(255 * progress)
            pos = (int(e["pos"][0]), int(e["pos"][1]))
            
            # Use gfxdraw for anti-aliased, alpha-blended circles
            if radius > 0:
                color = (*self.COLOR_EXPLOSION, alpha)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_base(self, base, color, label):
        pos = (int(base["pos"][0]), int(base["pos"][1]))
        size = self.BASE_SIZE
        rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Health bar
        if base["health"] > 0:
            self._render_health_bar(
                (pos[0], pos[1] - size/2 - 10),
                (size, 7),
                base["health"],
                self.BASE_HEALTH
            )
            health_text = self.font_small.render(f"{int(base['health'])}", True, self.COLOR_UI_TEXT)
            self.screen.blit(health_text, (pos[0] - health_text.get_width()/2, pos[1] - size/2 - 25))

    def _render_tank(self, tank, color):
        pos = (int(tank["pos"][0]), int(tank["pos"][1]))
        
        # Tank Body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TANK_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TANK_RADIUS, color)

        # Turret
        turret_len = self.TANK_RADIUS + 3
        turret_end = tank["pos"] + tank["dir"] * turret_len
        pygame.draw.line(self.screen, self.COLOR_WHITE, pos, (int(turret_end[0]), int(turret_end[1])), 4)
        
        # Health bar
        self._render_health_bar(
            (pos[0], pos[1] - self.TANK_RADIUS - 10),
            (self.TANK_RADIUS * 2, 5),
            tank["health"],
            self.TANK_HEALTH
        )
    
    def _render_health_bar(self, pos, size, health, max_health):
        x, y = pos
        w, h = size
        
        # Center the bar
        x -= w/2
        
        health_ratio = max(0, health / max_health)
        
        bg_rect = pygame.Rect(int(x), int(y), int(w), int(h))
        fg_rect = pygame.Rect(int(x), int(y), int(w * health_ratio), int(h))
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_left = self.MAX_STEPS - self.steps
        time_text = self.font_large.render(f"TIME: {steps_left // self.FPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            outcome_font = pygame.font.SysFont("sans", 72, bold=True)
            outcome_text = outcome_font.render(self.game_outcome, True, self.COLOR_WHITE if self.game_outcome == "VICTORY" else self.COLOR_ENEMY)
            text_rect = outcome_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_tank["health"],
            "enemy_health": self.enemy_tank["health"],
            "player_base_health": self.player_base["health"],
            "enemy_base_health": self.enemy_base["health"],
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override the dummy display with a real one for manual play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tank Arena")
    clock = pygame.time.Clock()
    
    done = False
    
    print(env.user_guide)
    print("Close the window to quit.")
    
    while not done:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()