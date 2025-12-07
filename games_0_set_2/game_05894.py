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
        "Controls: ↑↓←→ to move. Hold space to jump and shift to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Jump and attack monsters in a top-down arcade environment to defeat them all."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Game world
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    NUM_MONSTERS = 20

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
    COLOR_MONSTER = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)

    # Player physics
    PLAYER_SPEED = 4
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100
    JUMP_POWER = 10
    GRAVITY = 0.6
    
    # Attack
    ATTACK_COOLDOWN_FRAMES = 8
    PROJECTILE_SPEED = 12
    PROJECTILE_RADIUS = 4
    PROJECTILE_DAMAGE = 4

    # Monster
    MONSTER_SPEED = 1.2
    MONSTER_RADIUS = 10
    MONSTER_MAX_HEALTH = 10
    MONSTER_CONTACT_DAMAGE = 10

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.player = {}
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.attack_cooldown = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.attack_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.player = {
            "pos": pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2),
            "z": 0,
            "z_vel": 0,
            "is_jumping": False,
            "health": self.PLAYER_MAX_HEALTH
        }

        self.monsters = self._spawn_monsters(self.NUM_MONSTERS)
        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _spawn_monsters(self, num):
        monsters = []
        for _ in range(num):
            # Spawn away from the center
            side = self.np_random.integers(4)
            if side == 0: # top
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT * 0.2)
            elif side == 1: # bottom
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = self.np_random.uniform(self.SCREEN_HEIGHT * 0.8, self.SCREEN_HEIGHT)
            elif side == 2: # left
                x = self.np_random.uniform(0, self.SCREEN_WIDTH * 0.2)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            else: # right
                x = self.np_random.uniform(self.SCREEN_WIDTH * 0.8, self.SCREEN_WIDTH)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)

            monsters.append({
                "pos": pygame.Vector2(x, y),
                "health": self.MONSTER_MAX_HEALTH,
                "direction": self.np_random.choice([-1, 1])
            })
        return monsters

    def step(self, action):
        reward = -0.1  # Time penalty to encourage efficiency
        
        if not self.game_over:
            # --- Handle Actions ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Player Movement
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y = -1
            elif movement == 2: move_vec.y = 1
            elif movement == 3: move_vec.x = -1
            elif movement == 4: move_vec.x = 1
            if move_vec.length() > 0:
                move_vec.normalize_ip()
            self.player["pos"] += move_vec * self.PLAYER_SPEED
            self.player["pos"].x = np.clip(self.player["pos"].x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
            self.player["pos"].y = np.clip(self.player["pos"].y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

            # Player Jump (on key press)
            if space_held and not self.prev_space_held and not self.player["is_jumping"]:
                self.player["is_jumping"] = True
                self.player["z_vel"] = self.JUMP_POWER
                # sfx: jump

            # Player Attack (on key press with cooldown)
            if self.attack_cooldown > 0:
                self.attack_cooldown -= 1
            if shift_held and not self.prev_shift_held and self.attack_cooldown == 0 and self.monsters:
                self._fire_projectile()
                self.attack_cooldown = self.ATTACK_COOLDOWN_FRAMES

            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

            # --- Update Game State ---
            self._update_player_jump()
            self._update_monsters()
            reward += self._update_projectiles()
            self._update_particles()
            reward += self._handle_collisions()

        # --- Check Termination ---
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.player["health"] <= 0:
                reward -= 100 # Loss penalty
                # sfx: player_death
            elif not self.monsters:
                reward += 100 # Win bonus
                # sfx: victory
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_jump(self):
        if self.player["is_jumping"]:
            self.player["z_vel"] -= self.GRAVITY
            self.player["z"] += self.player["z_vel"]
            if self.player["z"] <= 0:
                self.player["z"] = 0
                self.player["z_vel"] = 0
                self.player["is_jumping"] = False
                # sfx: land
                self._create_particles(self.player["pos"], 10, self.COLOR_UI_TEXT, 0.5, 2, 10) # Landing dust

    def _update_monsters(self):
        for monster in self.monsters:
            monster["pos"].x += self.MONSTER_SPEED * monster["direction"]
            if monster["pos"].x <= self.MONSTER_RADIUS or monster["pos"].x >= self.SCREEN_WIDTH - self.MONSTER_RADIUS:
                monster["direction"] *= -1

    def _update_projectiles(self):
        hit_reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            if 0 < proj["pos"].x < self.SCREEN_WIDTH and 0 < proj["pos"].y < self.SCREEN_HEIGHT:
                hit = False
                for monster in self.monsters:
                    if proj["pos"].distance_to(monster["pos"]) < self.PROJECTILE_RADIUS + self.MONSTER_RADIUS:
                        monster["health"] -= self.PROJECTILE_DAMAGE
                        hit_reward += 1  # Reward for hitting
                        self._create_particles(proj["pos"], 15, self.COLOR_PROJECTILE, 2, 4, 15)
                        # sfx: monster_hit
                        hit = True
                        break
                if not hit:
                    projectiles_to_keep.append(proj)
        self.projectiles = projectiles_to_keep
        return hit_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _handle_collisions(self):
        # Player vs Monster
        collision_reward = 0
        if not self.player["is_jumping"]: # Can't be hit while in the air
            for monster in self.monsters:
                if self.player["pos"].distance_to(monster["pos"]) < self.PLAYER_RADIUS + self.MONSTER_RADIUS:
                    self.player["health"] -= self.MONSTER_CONTACT_DAMAGE
                    self._create_particles(self.player["pos"], 20, self.COLOR_MONSTER, 2, 5, 20)
                    # sfx: player_hurt
                    break # Only take damage from one monster per frame
        
        # Check for monster deaths
        monsters_alive = []
        for monster in self.monsters:
            if monster["health"] > 0:
                monsters_alive.append(monster)
            else:
                collision_reward += 10 # Reward for killing
                self.score += 100
                self._create_particles(monster["pos"], 50, self.COLOR_MONSTER, 1, 6, 30)
                # sfx: monster_death
        self.monsters = monsters_alive
        
        return collision_reward

    def _fire_projectile(self):
        # Find nearest monster
        if not self.monsters: return
        
        nearest_monster = min(self.monsters, key=lambda m: self.player["pos"].distance_squared_to(m["pos"]))
        direction = (nearest_monster["pos"] - self.player["pos"]).normalize()
        
        self.projectiles.append({
            "pos": pygame.Vector2(self.player["pos"]),
            "vel": direction * self.PROJECTILE_SPEED
        })
        # sfx: shoot

    def _check_termination(self):
        if self.player["health"] <= 0:
            self.player["health"] = 0
            return True
        if not self.monsters:
            return True
        return False

    def _get_observation(self):
        # --- Rendering ---
        self._render_background()
        self._render_monsters()
        self._render_player_shadow()
        self._render_projectiles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_monsters(self):
        for monster in self.monsters:
            pos = (int(monster["pos"].x), int(monster["pos"].y))
            # Pulsing effect
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(self.MONSTER_RADIUS * (0.9 + pulse * 0.2))
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_MONSTER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_MONSTER)

    def _render_player_shadow(self):
        shadow_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        z_factor = 1 - min(self.player["z"] / (self.JUMP_POWER**2 / (2 * self.GRAVITY) * 1.5), 1)
        radius = int(self.PLAYER_RADIUS * z_factor)
        pos = (int(self.player["pos"].x), int(self.player["pos"].y))
        
        if radius > 1:
            pygame.gfxdraw.filled_circle(shadow_surface, pos[0], pos[1], radius, self.COLOR_PLAYER_SHADOW)
        self.screen.blit(shadow_surface, (0, 0))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            lifespan_ratio = p["lifespan"] / p["max_lifespan"]
            radius = int(p["size"] * lifespan_ratio)
            if radius > 0:
                # Simple alpha blending by creating a temporary surface
                particle_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color = (*p["color"], int(255 * lifespan_ratio))
                pygame.draw.circle(particle_surface, color, (radius, radius), radius)
                self.screen.blit(particle_surface, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        z_factor = 1 + self.player["z"] / 80
        radius = int(self.PLAYER_RADIUS * z_factor)
        pos = (int(self.player["pos"].x), int(self.player["pos"].y - self.player["z"]))
        
        # Body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        # "Glow" effect
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 2, (*self.COLOR_PLAYER, 100))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Health Bar
        health_pct = self.player["health"] / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_pct), bar_height))
        
        # Game Over Text
        if self.game_over:
            if not self.monsters and self.player["health"] > 0:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_MONSTER
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)
    
    def _create_particles(self, pos, count, color, min_speed, max_speed, max_lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(max_lifespan // 2, max_lifespan)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "monsters_left": len(self.monsters)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example usage to run and visualize the game
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()

    # To render, we need a display
    try:
        pygame.display.set_caption("Arcade Jumper")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    except pygame.error:
        print("No display available, running headlessly.")
        screen = None

    terminated = False
    truncated = False
    running = True
    total_reward = 0

    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        action = [0, 0, 0] # Default action: do nothing
        
        if screen:
            # --- Human Controls ---
            movement = 0
            space_held = 0
            shift_held = 0
            
            keys = pygame.key.get_pressed()
            for key, move_action in key_map.items():
                if keys[key]:
                    movement = move_action
                    break # Prioritize first key found
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        total_reward = 0
        else: # No screen, just run with random actions
            action = env.action_space.sample()

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        elif screen: # If game over and we have a screen, wait for 'r' or quit
            pass
        else: # If game over and headless, reset automatically
             obs, info = env.reset()
             terminated = False
             truncated = False
             total_reward = 0


        if screen:
            # --- Rendering ---
            # The observation is already a rendered frame
            # We just need to convert it back to a Pygame surface to display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(30) # Control the frame rate

    env.close()