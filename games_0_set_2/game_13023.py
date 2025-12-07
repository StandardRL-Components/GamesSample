import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:48:31.933248
# Source Brief: brief_03023.md
# Brief Index: 3023
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Navigate a bubbly underwater world, trapping enemies in strategically placed air pockets
    using fluid dynamics and bubble propulsion.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate an underwater world, trapping enemies in bubbles you create. "
        "Pop the trapped enemies to score points, but watch your oxygen!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shoot a bubble to trap enemies."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 25, 60)
    COLOR_BG_ROCK = (40, 50, 80)
    COLOR_BG_CORAL = (70, 40, 60)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 150, 150)
    COLOR_BUBBLE = (220, 240, 255)
    COLOR_OXYGEN_BAR = (255, 220, 0)
    COLOR_OXYGEN_BAR_BG = (80, 80, 80)
    COLOR_TEXT = (255, 255, 255)
    COLOR_CURRENT_PARTICLE = (100, 120, 180, 100)

    # Player
    PLAYER_SPEED = 5.0
    PLAYER_SIZE = 12
    PLAYER_OXYGEN_MAX = 100.0
    PLAYER_OXYGEN_DEPLETION_RATE = 0.05  # Per step
    PLAYER_OXYGEN_HIT_LOSS = 15.0
    PLAYER_INVINCIBILITY_FRAMES = 60

    # Bubbles
    BUBBLE_SPEED = 2.0
    BUBBLE_COOLDOWN = 10  # Steps
    BUBBLE_MAX_SIZE = 15
    BUBBLE_LIFETIME = 180  # Steps
    BUBBLE_TRAP_LIFETIME = 300 # Steps

    # Enemies
    NUM_ENEMIES = 5
    ENEMY_SPEED_MIN = 1.0
    ENEMY_SPEED_MAX = 3.0
    ENEMY_SIZE = 10
    ENEMY_RESPAWN_TIME = 300 # Steps after being popped

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 40, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_facing_dir = None
        self.oxygen = None
        self.last_oxygen_level = None
        self.invincibility_timer = 0
        self.bubble_cooldown_timer = 0
        self.enemies = []
        self.bubbles = []
        self.particles = []
        self.background_elements = []
        self.water_current = None
        
        self._generate_background()
        # self.reset() is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_facing_dir = pygame.Vector2(0, -1) # Start facing up
        self.oxygen = self.PLAYER_OXYGEN_MAX
        self.last_oxygen_level = self.PLAYER_OXYGEN_MAX
        self.invincibility_timer = 0

        # Game elements state
        self.bubble_cooldown_timer = 0
        self.enemies = []
        self.bubbles = []
        self.particles = []
        self.defeated_enemies_queue = deque()

        for _ in range(self.NUM_ENEMIES):
            self._spawn_enemy()
            
        self._update_water_current()
        self._generate_background() # Re-gen for variety

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- ACTION HANDLING ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update Player Movement
        player_velocity = pygame.Vector2(0, 0)
        if movement == 1: player_velocity.y = -1 # Up
        elif movement == 2: player_velocity.y = 1  # Down
        elif movement == 3: player_velocity.x = -1 # Left
        elif movement == 4: player_velocity.x = 1  # Right
        
        if player_velocity.length() > 0:
            player_velocity.normalize_ip()
            self.player_facing_dir = player_velocity.copy()
        
        self.player_pos += player_velocity * self.PLAYER_SPEED
        self._keep_in_bounds(self.player_pos, self.PLAYER_SIZE)

        # 2. Handle Bubble Firing
        if self.bubble_cooldown_timer > 0:
            self.bubble_cooldown_timer -= 1

        if space_held and self.bubble_cooldown_timer == 0:
            # SFX: Bubble shoot sound
            self.bubbles.append({
                "pos": self.player_pos.copy(),
                "dir": self.player_facing_dir.copy(),
                "size": 5,
                "lifetime": self.BUBBLE_LIFETIME,
                "state": "rising", # 'rising' or 'trapped'
                "enemy_idx": -1,
            })
            self.bubble_cooldown_timer = self.BUBBLE_COOLDOWN

        # --- GAME LOGIC UPDATES ---
        
        # 3. Update Water Currents & Difficulty
        if self.steps % 250 == 0:
            self._update_water_current()
        
        current_enemy_speed = self.ENEMY_SPEED_MIN + (self.steps // 200) * 0.05
        current_enemy_speed = min(current_enemy_speed, self.ENEMY_SPEED_MAX)

        # 4. Update Bubbles
        for bubble in self.bubbles[:]:
            if bubble["state"] == "rising":
                bubble["pos"].y -= self.BUBBLE_SPEED
                bubble["pos"] += self.water_current * 0.5 # Influence from current
                bubble["size"] = min(self.BUBBLE_MAX_SIZE, bubble["size"] + 0.05)
            
            bubble["lifetime"] -= 1
            if bubble["lifetime"] <= 0 or bubble["pos"].y < -self.BUBBLE_MAX_SIZE:
                if bubble["state"] == "trapped": # Enemy escapes if bubble pops
                    if bubble["enemy_idx"] < len(self.enemies):
                        self.enemies[bubble["enemy_idx"]]["state"] = "free"
                self.bubbles.remove(bubble)
                # SFX: Gentle pop
                self._create_particles(bubble["pos"], 5, self.COLOR_BUBBLE)

        # 5. Update Enemies
        for i, enemy in enumerate(self.enemies):
            if enemy["state"] == "free":
                enemy["pos"] += enemy["dir"] * current_enemy_speed
                if enemy["pos"].x <= self.ENEMY_SIZE or enemy["pos"].x >= self.SCREEN_WIDTH - self.ENEMY_SIZE:
                    enemy["dir"].x *= -1
                if enemy["pos"].y <= self.ENEMY_SIZE or enemy["pos"].y >= self.SCREEN_HEIGHT - self.ENEMY_SIZE:
                    enemy["dir"].y *= -1
                self._keep_in_bounds(enemy["pos"], self.ENEMY_SIZE)
            elif enemy["state"] == "trapped":
                # Follow the bubble it's trapped in
                for bubble in self.bubbles:
                    if bubble["enemy_idx"] == i:
                        enemy["pos"] = bubble["pos"].copy()
                        break
        
        # 6. Update Particles & Invincibility
        for p in self.particles[:]:
            if p.get("type") != "current":
                p["pos"] += p["vel"]
                p["lifetime"] -= 1
                if p["lifetime"] <= 0:
                    self.particles.remove(p)
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        # 7. Respawn defeated enemies
        if self.defeated_enemies_queue:
            if self.steps >= self.defeated_enemies_queue[0][1]:
                idx, _ = self.defeated_enemies_queue.popleft()
                self._spawn_enemy(index_to_replace=idx)

        # --- COLLISION DETECTION & INTERACTIONS ---
        
        # Player vs. Free Enemy
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                if enemy["state"] == "free" and self.player_pos.distance_to(enemy["pos"]) < self.PLAYER_SIZE + self.ENEMY_SIZE:
                    # SFX: Player hit sound
                    self.oxygen -= self.PLAYER_OXYGEN_HIT_LOSS
                    reward -= 5.0
                    self.invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    self._create_particles(self.player_pos, 20, self.COLOR_ENEMY)
                    break # Only one hit per frame

        # Bubble vs. Free Enemy
        for bubble in self.bubbles:
            if bubble["state"] == "rising":
                for i, enemy in enumerate(self.enemies):
                    if enemy["state"] == "free" and bubble["pos"].distance_to(enemy["pos"]) < bubble["size"] + self.ENEMY_SIZE:
                        # SFX: Enemy trapped sound
                        enemy["state"] = "trapped"
                        bubble["state"] = "trapped"
                        bubble["enemy_idx"] = i
                        bubble["lifetime"] = self.BUBBLE_TRAP_LIFETIME
                        reward += 0.1
                        self.score += 10
                        break # Trap one enemy per bubble

        # Player vs. Trapped Bubble (Pop)
        for bubble in self.bubbles[:]:
            if bubble["state"] == "trapped" and self.player_pos.distance_to(bubble["pos"]) < self.PLAYER_SIZE + bubble["size"]:
                # SFX: Satisfying pop sound
                enemy_idx = bubble["enemy_idx"]
                if enemy_idx < len(self.enemies):
                    self.enemies[enemy_idx]["state"] = "defeated"
                    self.defeated_enemies_queue.append((enemy_idx, self.steps + self.ENEMY_RESPAWN_TIME))
                
                self._create_particles(bubble["pos"], 30, self.COLOR_BUBBLE)
                self.bubbles.remove(bubble)
                
                reward += 1.0
                self.score += 100
                break

        # --- OXYGEN & TERMINATION ---
        
        # 8. Update Oxygen
        self.oxygen -= self.PLAYER_OXYGEN_DEPLETION_RATE
        oxygen_depleted_percent = self.last_oxygen_level - self.oxygen
        if oxygen_depleted_percent > 0:
            reward -= oxygen_depleted_percent * 0.1 # -0.1 per 1% loss
        self.last_oxygen_level = self.oxygen
        
        # 9. Check Termination Conditions
        terminated = False
        
        if self.oxygen <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            reward += 100 # Survival bonus
            terminated = True
            self.game_over = True
            
        truncated = False # This env does not truncate based on time limit, it terminates.
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "oxygen": self.oxygen,
            "enemies_left": sum(1 for e in self.enemies if e["state"] != "defeated"),
        }
    
    def close(self):
        pygame.quit()

    # --- Helper & Rendering Methods ---

    def _generate_background(self):
        self.background_elements = []
        for _ in range(20): # Rocks
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(100, self.SCREEN_HEIGHT))
            radius = random.randint(20, 80)
            self.background_elements.append(("rock", pos, radius))
        for _ in range(15): # Coral
            pos = (random.randint(0, self.SCREEN_WIDTH), random.randint(200, self.SCREEN_HEIGHT))
            height = random.randint(30, 100)
            self.background_elements.append(("coral", pos, height))
        
        # Clear and recreate current particles
        self.particles = [p for p in self.particles if p.get("type") != "current"]
        for _ in range(300): # Current particles
            pos = pygame.Vector2(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT))
            self.particles.append({"pos": pos, "vel": pygame.Vector2(0,0), "lifetime": 99999, "type": "current"})

    def _render_background(self):
        for type, pos, val in self.background_elements:
            if type == "rock":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], val, self.COLOR_BG_ROCK)
            elif type == "coral":
                pygame.draw.rect(self.screen, self.COLOR_BG_CORAL, (pos[0], pos[1] - val, 10, val))

    def _render_game(self):
        # Particles
        for p in self.particles:
            if p.get("type") == "current":
                p["pos"] += self.water_current * 0.2
                if p["pos"].x > self.SCREEN_WIDTH: p["pos"].x = 0
                if p["pos"].x < 0: p["pos"].x = self.SCREEN_WIDTH
                if p["pos"].y > self.SCREEN_HEIGHT: p["pos"].y = 0
                if p["pos"].y < 0: p["pos"].y = self.SCREEN_HEIGHT
                pygame.gfxdraw.pixel(self.screen, int(p["pos"].x), int(p["pos"].y), self.COLOR_CURRENT_PARTICLE)
            else:
                alpha = max(0, 255 * (p["lifetime"] / p["initial_lifetime"]))
                color = (*p["color"], alpha)
                s = pygame.Surface((int(p["lifetime"] * 0.2) * 2, int(p["lifetime"] * 0.2) * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (int(p["lifetime"] * 0.1), int(p["lifetime"] * 0.1)), int(p["lifetime"] * 0.1))
                self.screen.blit(s, (int(p["pos"].x - p["lifetime"]*0.1), int(p["pos"].y - p["lifetime"]*0.1)))

        # Bubbles
        for bubble in self.bubbles:
            pos = (int(bubble["pos"].x), int(bubble["pos"].y))
            size = int(bubble["size"])
            # Draw semi-transparent bubble
            surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*self.COLOR_BUBBLE, 100), (size, size), size)
            pygame.draw.circle(surface, (*self.COLOR_BUBBLE, 200), (size, size), size, 2)
            self.screen.blit(surface, (pos[0] - size, pos[1] - size))

        # Enemies
        for enemy in self.enemies:
            if enemy["state"] != "defeated":
                pos = (int(enemy["pos"].x), int(enemy["pos"].y))
                size = self.ENEMY_SIZE
                glow_size = int(size * (1.5 + 0.2 * math.sin(self.steps * 0.1)))
                
                # Using SRCALPHA surfaces for smooth anti-aliased glow
                glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_ENEMY_GLOW, 50))
                pygame.gfxdraw.aacircle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_ENEMY_GLOW, 100))
                self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size))

                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
                if enemy["state"] == "trapped": # Add wobble
                    pos = (pos[0] + int(2*math.sin(self.steps*0.3)), pos[1] + int(2*math.cos(self.steps*0.3)))

        # Player
        if self.invincibility_timer > 0 and self.steps % 10 < 5:
            pass # Flicker effect when invincible
        else:
            pos = (int(self.player_pos.x), int(self.player_pos.y))
            size = self.PLAYER_SIZE
            glow_size = int(size * 2.5)

            glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, 50))
            pygame.gfxdraw.aacircle(glow_surf, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, 100))
            self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size))

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
            # Facing direction indicator
            dir_end = self.player_pos + self.player_facing_dir * (size + 3)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, pos, (int(dir_end.x), int(dir_end.y)), 2)

    def _render_ui(self):
        # Oxygen Bar
        bar_width = 200
        bar_height = 20
        oxygen_percent = max(0, self.oxygen / self.PLAYER_OXYGEN_MAX)
        current_width = int(bar_width * oxygen_percent)
        pygame.draw.rect(self.screen, self.COLOR_OXYGEN_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_OXYGEN_BAR, (10, 10, current_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, bar_width, bar_height), 1)

        # Score and Info Text
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        enemies_left = sum(1 for e in self.enemies if e["state"] != "defeated")
        enemies_text = self.font_main.render(f"ENEMIES: {enemies_left}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.SCREEN_WIDTH - enemies_text.get_width() - 10, 35))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "SURVIVED!" if self.steps >= self.MAX_STEPS and self.oxygen > 0 else "OUT OF OXYGEN"
            end_text = self.font_title.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
            self.screen.blit(final_score_text, score_rect)

    def _spawn_enemy(self, index_to_replace=None):
        enemy = {
            "pos": pygame.Vector2(random.randint(50, self.SCREEN_WIDTH - 50), random.randint(50, self.SCREEN_HEIGHT - 150)),
            "dir": pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize(),
            "state": "free", # 'free', 'trapped', 'defeated'
        }
        if index_to_replace is not None:
            self.enemies[index_to_replace] = enemy
        else:
            self.enemies.append(enemy)

    def _keep_in_bounds(self, vec, margin):
        vec.x = max(margin, min(self.SCREEN_WIDTH - margin, vec.x))
        vec.y = max(margin, min(self.SCREEN_HEIGHT - margin, vec.y))

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            lifetime = random.randint(20, 40)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                "lifetime": lifetime,
                "initial_lifetime": lifetime,
                "color": color
            })

    def _update_water_current(self):
        angle = random.uniform(0, 2 * math.pi)
        strength = random.uniform(0.5, 1.5)
        self.water_current = pygame.Vector2(math.cos(angle), math.sin(angle)) * strength


if __name__ == '__main__':
    # This block allows you to play the game manually
    # You must have pygame installed: pip install pygame
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bubble Trouble Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
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
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.GAME_FPS)
        
    env.close()