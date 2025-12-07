
# Generated: 2025-08-28T05:14:18.848133
# Source Brief: brief_05510.md
# Brief Index: 5510

        
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
        "Controls: Arrow keys to move, Space to fire."
    )

    # User-facing description of the game
    game_description = (
        "Defend Earth from a descending alien horde in a retro-inspired top-down arcade shooter."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.TOTAL_STAGES = 3

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_THRUSTER = (255, 100, 50)
        self.COLOR_ALIEN_A = (255, 50, 150)
        self.COLOR_ALIEN_B = (200, 50, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ALIEN_PROJ = (255, 50, 50)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 0
        self.player_pos = None
        self.player_lives = 0
        self.player_speed = 0
        self.player_fire_cooldown = 0
        self.last_space_held = False
        self.aliens = []
        self.alien_block_pos = None
        self.alien_direction = 1
        self.alien_move_timer = 0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.player_hit_timer = 0
        self.reward = 0

        self.reset()
        
        # Self-check validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_speed = 8
        self.player_fire_cooldown = 0
        self.last_space_held = False
        self.aliens = []
        self.alien_block_pos = [50, 50]
        self.alien_direction = 1
        self.alien_move_timer = 0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.player_hit_timer = 0
        self.reward = 0

        self._create_stars(200)
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward = 0.1  # Survival reward
        self.steps += 1

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Calculate distance to nearest alien before moving
        dist_before = self._get_dist_to_nearest_alien()

        self._handle_input(movement, space_held)
        
        # Calculate distance to nearest alien after moving
        dist_after = self._get_dist_to_nearest_alien()
        
        if dist_after > dist_before:
            self.reward -= 0.02

        self._update_entities()
        self._handle_collisions()
        self._cleanup()

        if not self.aliens:
            self._advance_stage()

        terminated = self.game_over or self.steps >= self.MAX_STEPS

        # Final reward for winning the game
        if self.stage > self.TOTAL_STAGES and not self.game_over:
            self.reward += 500
            self.game_over = True
            terminated = True
        
        final_reward = self.reward

        return (
            self._get_observation(),
            final_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_background()
        self._render_game()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    # --- Helper Methods ---

    def _setup_stage(self):
        self.aliens.clear()
        self.player_projectiles.clear()
        self.alien_projectiles.clear()
        self.alien_block_pos = [50, 50]
        
        rows, cols = 3, 8
        for row in range(rows):
            for col in range(cols):
                alien_type = self.COLOR_ALIEN_A if row % 2 == 0 else self.COLOR_ALIEN_B
                self.aliens.append({
                    "rel_pos": [col * 50, row * 40],
                    "color": alien_type,
                    "alive": True
                })

    def _advance_stage(self):
        if self.stage <= self.TOTAL_STAGES:
            self.reward += 100
            self.stage += 1
            if self.stage <= self.TOTAL_STAGES:
                self._setup_stage()
                self._create_particles(self.WIDTH // 2, self.HEIGHT // 2, 50, self.COLOR_TEXT)
        else:
            self.game_over = True

    def _create_stars(self, count):
        self.stars = [
            {
                "pos": [random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)],
                "speed": random.uniform(0.5, 1.5),
                "size": random.randint(1, 2)
            }
            for _ in range(count)
        ]

    def _get_dist_to_nearest_alien(self):
        if not self.aliens:
            return 0
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for alien in self.aliens:
            if alien["alive"]:
                alien_x = self.alien_block_pos[0] + alien["rel_pos"][0]
                alien_y = self.alien_block_pos[1] + alien["rel_pos"][1]
                dist = math.hypot(player_x - alien_x, player_y - alien_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.player_speed
        elif movement == 2: self.player_pos[1] += self.player_speed
        elif movement == 3: self.player_pos[0] -= self.player_speed
        elif movement == 4: self.player_pos[0] += self.player_speed

        # Screen wrapping for player
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

        # Firing on button press (rising edge)
        if space_held and not self.last_space_held and self.player_fire_cooldown <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append({"pos": self.player_pos.copy(), "speed": -15})
            self.player_fire_cooldown = 8 # 8 frames cooldown

        self.last_space_held = space_held
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

    def _update_entities(self):
        # Player hit timer
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1
            
        # Update stars
        for star in self.stars:
            star["pos"][1] += star["speed"]
            if star["pos"][1] > self.HEIGHT:
                star["pos"] = [random.randint(0, self.WIDTH), 0]

        # Update player projectiles
        for p in self.player_projectiles:
            p["pos"][1] += p["speed"]
            p["pos"][0] %= self.WIDTH
            p["pos"][1] %= self.HEIGHT

        # Update alien projectiles
        alien_proj_speed = 5 + (self.stage - 1) * 0.5
        for p in self.alien_projectiles:
            p["pos"][1] += alien_proj_speed
            p["pos"][0] %= self.WIDTH
            p["pos"][1] %= self.HEIGHT
            
        # Update alien block movement
        move_speed = 1.0 + self.stage * 0.5
        self.alien_block_pos[0] += self.alien_direction * move_speed
        
        if self.alien_direction == 1 and self.alien_block_pos[0] > self.WIDTH - 8 * 50 - 20:
            self.alien_direction = -1
            self.alien_block_pos[1] += 10
        elif self.alien_direction == -1 and self.alien_block_pos[0] < 20:
            self.alien_direction = 1
            self.alien_block_pos[1] += 10
            
        # Alien firing
        fire_prob = (0.005 + (self.stage - 1) * 0.005)
        if self.aliens and random.random() < fire_prob:
            firing_alien = random.choice([a for a in self.aliens if a["alive"]])
            if firing_alien:
                # sfx: alien_shoot.wav
                fire_pos = [self.alien_block_pos[0] + firing_alien["rel_pos"][0] + 10,
                            self.alien_block_pos[1] + firing_alien["rel_pos"][1] + 20]
                self.alien_projectiles.append({"pos": fire_pos})

        # Update particles
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)

        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            p_rect = pygame.Rect(p["pos"][0] - 2, p["pos"][1] - 8, 4, 16)
            for alien in self.aliens:
                if alien["alive"]:
                    alien_rect = pygame.Rect(
                        self.alien_block_pos[0] + alien["rel_pos"][0],
                        self.alien_block_pos[1] + alien["rel_pos"][1],
                        25, 20
                    )
                    if p_rect.colliderect(alien_rect):
                        # sfx: alien_explosion.wav
                        alien["alive"] = False
                        self.score += 10
                        self.reward += 10
                        self._create_particles(alien_rect.centerx, alien_rect.centery, 20, alien["color"])
                        if p in self.player_projectiles: self.player_projectiles.remove(p)
                        break

        # Alien projectiles vs Player
        if self.player_hit_timer <= 0:
            for p in self.alien_projectiles[:]:
                p_rect = pygame.Rect(p["pos"][0] - 3, p["pos"][1] - 3, 6, 6)
                if player_rect.colliderect(p_rect):
                    # sfx: player_hit.wav
                    self.player_lives -= 1
                    self.reward -= 5
                    self.player_hit_timer = self.FPS * 2  # 2 seconds of invincibility
                    self._create_particles(player_rect.centerx, player_rect.centery, 30, self.COLOR_PLAYER)
                    if p in self.alien_projectiles: self.alien_projectiles.remove(p)
                    if self.player_lives <= 0:
                        self.game_over = True
                    break

    def _cleanup(self):
        self.player_projectiles = [p for p in self.player_projectiles if 0 <= p["pos"][1] <= self.HEIGHT]
        self.alien_projectiles = [p for p in self.alien_projectiles if 0 <= p["pos"][1] <= self.HEIGHT]
        self.aliens = [a for a in self.aliens if a["alive"]]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
    
    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": random.randint(15, 30),
                "color": random.choice(self.COLOR_EXPLOSION) if color != self.COLOR_TEXT else self.COLOR_TEXT,
                "size": random.randint(1, 4)
            })

    # --- Rendering Methods ---

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(star["pos"][0]), int(star["pos"][1])), star["size"])

    def _render_game(self):
        # Render aliens
        for alien in self.aliens:
            if alien["alive"]:
                x = self.alien_block_pos[0] + alien["rel_pos"][0]
                y = self.alien_block_pos[1] + alien["rel_pos"][1]
                anim_offset = math.sin(self.steps * 0.2) * 3
                pygame.gfxdraw.box(self.screen, pygame.Rect(int(x), int(y + anim_offset), 25, 20), alien["color"])

        # Render player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (p["pos"][0] - 2, p["pos"][1] - 8, 4, 16))

        # Render alien projectiles
        for p in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 4, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 4, self.COLOR_ALIEN_PROJ)

        # Render player
        if self.player_lives > 0:
            is_invincible = self.player_hit_timer > 0
            if not (is_invincible and (self.steps // 3) % 2):
                px, py = int(self.player_pos[0]), int(self.player_pos[1])
                # Ship body
                points = [(px, py - 12), (px - 10, py + 8), (px + 10, py + 8)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
                # Thruster
                thruster_size = random.randint(4, 8)
                t_points = [(px, py + 8), (px - 4, py + 8 + thruster_size), (px + 4, py + 8 + thruster_size)]
                pygame.gfxdraw.filled_polygon(self.screen, t_points, self.COLOR_PLAYER_THRUSTER)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player_lives):
            px, py = self.WIDTH - 20 - (i * 25), 20
            points = [(px, py - 8), (px - 6, py + 5), (px + 6, py + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "VICTORY" if self.stage > self.TOTAL_STAGES else "GAME OVER"
            color = self.COLOR_PLAYER if msg == "VICTORY" else self.COLOR_ALIEN_PROJ
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
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

# Example usage (for testing)
if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    
    # To run headlessly and test API
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated:
    #         print(f"Episode finished. Info: {info}")
    #         env.reset()
    # env.close()
    
    # To run with visualization
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Horde Defender")
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)
        
    env.close()