
# Generated: 2025-08-28T02:22:15.941437
# Source Brief: brief_04428.md
# Brief Index: 4428

        
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

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to attack."
    )

    # User-facing description of the game
    game_description = (
        "Defeat waves of procedurally generated monsters in a side-scrolling arcade brawler using jumps and attacks."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PLATFORM = (87, 51, 30)
    COLOR_PLAYER = (50, 205, 50)
    COLOR_PLAYER_ATTACK = (255, 255, 100)
    COLOR_MONSTER_A = (180, 50, 90)
    COLOR_MONSTER_B = (150, 40, 120)
    COLOR_HEALTH_GREEN = (0, 255, 0)
    COLOR_HEALTH_RED = (255, 0, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    
    # Physics & Gameplay
    GRAVITY = 0.8
    PLAYER_SPEED = 6
    JUMP_STRENGTH = 15
    PLAYER_MAX_HEALTH = 100
    MONSTER_MAX_HEALTH = 20
    MONSTER_CONTACT_DAMAGE = 5
    PLAYER_ATTACK_DAMAGE = 5
    ATTACK_DURATION_FRAMES = 4
    ATTACK_COOLDOWN_FRAMES = 12
    INITIAL_MONSTER_SPEED = 0.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.monster_base_speed = 0.0
        self.game_over = False
        
        self.player = {}
        self.monsters = []
        self.particles = []
        self.attack_state = {}
        
        self.platform_y = self.SCREEN_HEIGHT * 4 // 5

        # Initialize state
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.monster_base_speed = self.INITIAL_MONSTER_SPEED
        self.game_over = False
        
        player_width, player_height = 30, 40
        self.player = {
            "rect": pygame.Rect(self.SCREEN_WIDTH // 2 - player_width // 2, self.platform_y - player_height, player_width, player_height),
            "vel_y": 0,
            "health": self.PLAYER_MAX_HEALTH,
            "on_ground": True,
            "facing_right": True
        }
        
        self.attack_state = {
            "active": False,
            "timer": 0,
            "cooldown": 0,
            "rect": pygame.Rect(0, 0, 0, 0),
            "hit_monsters": set()
        }
        
        self.monsters = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        reward += self._update_player()
        reward += self._update_monsters()
        reward += self._update_attack()
        self._update_particles()
        
        # --- Check for wave clear ---
        if not self.monsters:
            reward += 10 # Wave clear bonus
            self.wave += 1
            self.monster_base_speed += 0.05
            self._spawn_wave()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 10 # Death penalty
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.auto_advance:
            self.clock.tick(self.FPS)
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player["rect"].x -= self.PLAYER_SPEED
            self.player["facing_right"] = False
        elif movement == 4: # Right
            self.player["rect"].x += self.PLAYER_SPEED
            self.player["facing_right"] = True

        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel_y"] = -self.JUMP_STRENGTH
            self.player["on_ground"] = False
            # sfx: jump

        # Attacking
        if space_held and self.attack_state["cooldown"] <= 0:
            self.attack_state["active"] = True
            self.attack_state["timer"] = self.ATTACK_DURATION_FRAMES
            self.attack_state["cooldown"] = self.ATTACK_COOLDOWN_FRAMES
            self.attack_state["hit_monsters"].clear()
            # sfx: attack_swing

    def _update_player(self):
        # Apply gravity
        self.player["vel_y"] += self.GRAVITY
        self.player["rect"].y += int(self.player["vel_y"])
        
        # Ground collision
        if self.player["rect"].bottom >= self.platform_y:
            self.player["rect"].bottom = self.platform_y
            self.player["vel_y"] = 0
            if not self.player["on_ground"]:
                # Landing effect
                for _ in range(5):
                    self._create_particles(self.player["rect"].midbottom, 1, color=(139, 69, 19))
                self.player["on_ground"] = True
        
        # Screen boundaries
        self.player["rect"].left = max(0, self.player["rect"].left)
        self.player["rect"].right = min(self.SCREEN_WIDTH, self.player["rect"].right)
        
        return 0 # No intrinsic reward for player movement

    def _update_monsters(self):
        reward = 0
        for monster in self.monsters[:]:
            # Sinusoidal movement towards player
            target_x = self.player["rect"].centerx
            direction = 1 if target_x > monster["rect"].centerx else -1
            
            monster["rect"].x += direction * self.monster_base_speed
            
            y_offset = math.sin(self.steps * monster["anim_speed"] + monster["anim_offset"]) * monster["anim_amplitude"]
            monster["rect"].bottom = self.platform_y + y_offset

            # Player collision
            if self.player["rect"].colliderect(monster["rect"]):
                damage = self.MONSTER_CONTACT_DAMAGE
                self.player["health"] -= damage
                reward -= 0.1 * damage
                self._create_particles(self.player["rect"].center, 5, color=self.COLOR_HEALTH_RED)
                # sfx: player_hurt
        
        self.player["health"] = max(0, self.player["health"])
        return reward
    
    def _update_attack(self):
        reward = 0
        if self.attack_state["cooldown"] > 0:
            self.attack_state["cooldown"] -= 1

        if not self.attack_state["active"]:
            return 0
        
        self.attack_state["timer"] -= 1
        if self.attack_state["timer"] <= 0:
            self.attack_state["active"] = False
            return 0

        # Define attack hitbox
        attack_width, attack_height = 45, 30
        if self.player["facing_right"]:
            self.attack_state["rect"] = pygame.Rect(self.player["rect"].right, self.player["rect"].centery - attack_height // 2, attack_width, attack_height)
        else:
            self.attack_state["rect"] = pygame.Rect(self.player["rect"].left - attack_width, self.player["rect"].centery - attack_height // 2, attack_width, attack_height)
        
        # Check for hits
        for i, monster in enumerate(self.monsters):
            if i not in self.attack_state["hit_monsters"] and self.attack_state["rect"].colliderect(monster["rect"]):
                self.attack_state["hit_monsters"].add(i)
                damage = self.PLAYER_ATTACK_DAMAGE
                monster["health"] -= damage
                reward += 0.1 * damage
                self._create_particles(monster["rect"].center, 5, color=self.COLOR_PLAYER_ATTACK)
                # sfx: monster_hit
                
                if monster["health"] <= 0:
                    reward += 1 # Monster defeat bonus
                    self.score += 1
                    self.monsters.remove(monster)
                    self._create_particles(monster["rect"].center, 20, color=self.COLOR_MONSTER_A)
                    # sfx: monster_death
        return reward

    def _spawn_wave(self):
        self.wave += 1
        num_monsters = 5
        for i in range(num_monsters):
            monster_width, monster_height = 25, 25
            x_pos = random.choice([random.randint(-100, -30), random.randint(self.SCREEN_WIDTH + 30, self.SCREEN_WIDTH + 100)])
            
            self.monsters.append({
                "rect": pygame.Rect(x_pos, self.platform_y - monster_height, monster_width, monster_height),
                "health": self.MONSTER_MAX_HEALTH,
                "anim_offset": random.uniform(0, 2 * math.pi),
                "anim_speed": random.uniform(0.05, 0.1),
                "anim_amplitude": random.uniform(5, 15),
                "color": random.choice([self.COLOR_MONSTER_A, self.COLOR_MONSTER_B])
            })
            
    def _create_particles(self, pos, count, color=(255, 255, 255)):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [random.uniform(-2, 2), random.uniform(-3, 1)],
                "life": random.randint(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity on particles
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0, self.platform_y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.platform_y))

        # Draw particles
        for p in self.particles:
            size = max(1, int(p["life"] / 4))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # Draw monsters
        for monster in self.monsters:
            pygame.draw.rect(self.screen, monster["color"], monster["rect"])
            self._render_health_bar(monster["rect"].top - 10, monster["rect"].centerx, monster["health"], self.MONSTER_MAX_HEALTH)
        
        # Draw player
        player_color = self.COLOR_PLAYER_ATTACK if self.attack_state["active"] else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, player_color, self.player["rect"])

        # Draw attack slash
        if self.attack_state["active"]:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_ATTACK, self.attack_state["rect"], 1)

    def _render_ui(self):
        # Player Health Bar
        self._render_health_bar(15, self.player["rect"].centerx, self.player["health"], self.PLAYER_MAX_HEALTH, width=100)
        
        # Score Text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave Text
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

    def _render_health_bar(self, y, center_x, health, max_health, width=40):
        if health < 0: health = 0
        health_ratio = health / max_health
        bar_height = 5
        
        bg_rect = pygame.Rect(center_x - width // 2, y, width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, bg_rect)
        
        fg_width = int(width * health_ratio)
        fg_rect = pygame.Rect(center_x - width // 2, y, fg_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player["health"],
            "monsters_left": len(self.monsters),
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # key: (movement, space, shift)
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }
    
    # Main game loop
    total_reward = 0
    while not done:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Default action is no-op
        action = [0, 0, 0] 
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        
        # Other actions
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the game screen
        # The environment's observation is already the rendered screen
        # So we just need to display it
        display_surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Arcade Brawler")
        
        # Convert observation back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Episode Finished. Total Reward: {total_reward}, Final Info: {info}")
            # Optional: Add a delay or wait for key press before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

    env.close()