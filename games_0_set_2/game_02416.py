
# Generated: 2025-08-27T20:18:00.034976
# Source Brief: brief_02416.md
# Brief Index: 2416

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to attack nearby spiders. "
        "Collect 5 blue serum samples to win. Avoid the red spiders!"
    )

    game_description = (
        "Survive waves of spiders in a dark arena. Collect all 5 serum "
        "samples to win, but watch your health! The longer you survive, the "
        "more spiders will appear."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_ARENA_WALL = (40, 45, 55)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 40)
    COLOR_SPIDER = (255, 60, 60)
    COLOR_SPIDER_GLOW = (255, 60, 60, 50)
    COLOR_SERUM = (100, 200, 255)
    COLOR_SERUM_GLOW = (100, 200, 255, 60)
    COLOR_ATTACK = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (80, 40, 40)
    COLOR_HEALTH_BAR_FG = (100, 220, 100)

    # Game parameters
    PLAYER_RADIUS = 10
    PLAYER_SPEED = 4.0
    PLAYER_ATTACK_RADIUS = 60
    PLAYER_ATTACK_COOLDOWN = 15  # frames
    PLAYER_ATTACK_DURATION = 4 # frames

    SPIDER_RADIUS = 6
    SPIDER_SPEED = 1.5
    SPIDER_DAMAGE = 20
    MAX_SPIDERS = 20
    INITIAL_SPAWN_RATE = 0.5  # spiders per second
    SPAWN_RATE_INCREASE = 0.005 # per second

    SERUM_RADIUS = 8
    SERUM_COUNT = 5

    ARENA_RADIUS = 190
    ARENA_CENTER = (WIDTH // 2, HEIGHT // 2)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_health = 100
        self.spiders = []
        self.serums = []
        self.particles = []
        self.attack_timer = 0
        self.attack_cooldown_timer = 0
        self.serum_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_hit_timer = 0

        self.reset()
        # self.validate_implementation() # Optional: Call for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.ARENA_CENTER)
        self.player_health = 100
        self.serum_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.spiders.clear()
        self.serums.clear()
        self.particles.clear()
        
        self.attack_timer = 0
        self.attack_cooldown_timer = 0
        self.last_space_held = False
        self.last_hit_timer = 0

        self._spawn_serums()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing to encourage efficiency

        # --- Store pre-step state for reward calculation ---
        dist_to_serum_before = self._get_dist_to_closest_serum()

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_movement(movement)
        self._handle_player_attack(space_held)

        # --- Update Game State ---
        self._update_spiders()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Post-step state for reward calculation ---
        dist_to_serum_after = self._get_dist_to_closest_serum()
        if dist_to_serum_after < dist_to_serum_before:
            reward += 0.02 # Reward for moving towards objective

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.serum_collected >= self.SERUM_COUNT:
                reward += 100  # Win bonus
                self.score += 1000
            if self.player_health <= 0:
                reward -= 100  # Death penalty
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_dist_to_closest_serum(self):
        if not self.serums:
            return float('inf')
        return min(self.player_pos.distance_to(s) for s in self.serums)

    def _handle_player_movement(self, movement):
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Keep player within arena
        dist_from_center = self.player_pos.distance_to(self.ARENA_CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.PLAYER_RADIUS:
            vec_to_center = pygame.math.Vector2(self.ARENA_CENTER) - self.player_pos
            vec_to_center.scale_to_length(dist_from_center - (self.ARENA_RADIUS - self.PLAYER_RADIUS))
            self.player_pos += vec_to_center

    def _handle_player_attack(self, space_held):
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1
        
        if space_held and self.attack_cooldown_timer == 0:
            # SFX: PlayerAttack.wav
            self.attack_timer = self.PLAYER_ATTACK_DURATION
            self.attack_cooldown_timer = self.PLAYER_ATTACK_COOLDOWN

        self.last_space_held = space_held

    def _update_spiders(self):
        # Spawn new spiders
        spawn_chance = (self.INITIAL_SPAWN_RATE + self.steps / self.FPS * self.SPAWN_RATE_INCREASE) / self.FPS
        if self.np_random.random() < spawn_chance and len(self.spiders) < self.MAX_SPIDERS:
            self._spawn_spider()

        # Move existing spiders
        for spider in self.spiders:
            direction = (self.player_pos - spider['pos']).normalize()
            spider['pos'] += direction * self.SPIDER_SPEED

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - p['decay'])

    def _handle_collisions(self):
        reward = 0
        
        # Player attack vs Spiders
        if self.attack_timer > 0:
            self.attack_timer -= 1
            killed_spiders = []
            for spider in self.spiders:
                if self.player_pos.distance_to(spider['pos']) < self.PLAYER_ATTACK_RADIUS:
                    killed_spiders.append(spider)
                    reward += 1.0  # Reward for killing a spider
                    self.score += 100
                    # SFX: SpiderDeath.wav
                    self._create_particles(spider['pos'], 20, self.COLOR_SPIDER, 3.0, 15)
            self.spiders = [s for s in self.spiders if s not in killed_spiders]

        # Player vs Spiders
        if self.last_hit_timer > 0:
            self.last_hit_timer -= 1
        
        for spider in self.spiders:
            if self.player_pos.distance_to(spider['pos']) < self.PLAYER_RADIUS + self.SPIDER_RADIUS and self.last_hit_timer == 0:
                self.player_health -= self.SPIDER_DAMAGE
                self.last_hit_timer = self.FPS // 2 # 0.5 sec invulnerability
                self.score -= 50
                # SFX: PlayerHit.wav
                self._create_particles(self.player_pos, 15, self.COLOR_PLAYER, 2.0, 20)
                break # Only take damage from one spider per frame

        # Player vs Serums
        collected_serums = []
        for serum_pos in self.serums:
            if self.player_pos.distance_to(serum_pos) < self.PLAYER_RADIUS + self.SERUM_RADIUS:
                collected_serums.append(serum_pos)
                self.serum_collected += 1
                reward += 5.0 # Reward for collecting serum
                self.score += 500
                # SFX: SerumCollect.wav
                self._create_particles(serum_pos, 30, self.COLOR_SERUM, 4.0, 25, 'implode')
        self.serums = [s for s in self.serums if s not in collected_serums]
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0 or self.serum_collected >= self.SERUM_COUNT or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_serums(self):
        self.serums.clear()
        for _ in range(self.SERUM_COUNT):
            while True:
                angle = self.np_random.uniform(0, 2 * math.pi)
                # Spawn further from center to make it challenging
                radius = self.np_random.uniform(self.ARENA_RADIUS * 0.2, self.ARENA_RADIUS * 0.95)
                pos = pygame.math.Vector2(
                    self.ARENA_CENTER[0] + radius * math.cos(angle),
                    self.ARENA_CENTER[1] + radius * math.sin(angle)
                )
                # Ensure it doesn't spawn too close to player or other serums
                if pos.distance_to(self.player_pos) > 50 and all(pos.distance_to(s) > 30 for s in self.serums):
                    self.serums.append(pos)
                    break

    def _spawn_spider(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        pos = pygame.math.Vector2(
            self.ARENA_CENTER[0] + self.ARENA_RADIUS * math.cos(angle),
            self.ARENA_CENTER[1] + self.ARENA_RADIUS * math.sin(angle)
        )
        self.spiders.append({'pos': pos})

    def _create_particles(self, pos, count, color, max_speed, life, p_type='explode'):
        for _ in range(count):
            if p_type == 'explode':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, max_speed)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            else: # implode
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, max_speed)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * -speed
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'color': color,
                'radius': self.np_random.uniform(2, 5),
                'decay': 0.1
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena wall
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]), self.ARENA_RADIUS, self.COLOR_ARENA_WALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]), self.ARENA_RADIUS-1, self.COLOR_ARENA_WALL)

        # Serums
        for pos in self.serums:
            x, y = int(pos.x), int(pos.y)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.SERUM_RADIUS + 4, self.COLOR_SERUM_GLOW)
            pygame.gfxdraw.box(self.screen, (x - self.SERUM_RADIUS, y - self.SERUM_RADIUS, self.SERUM_RADIUS*2, self.SERUM_RADIUS*2), self.COLOR_SERUM)
        
        # Spiders
        for spider in self.spiders:
            x, y = int(spider['pos'].x), int(spider['pos'].y)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.SPIDER_RADIUS + 5, self.COLOR_SPIDER_GLOW)
            # Draw triangle for spider body
            angle_to_player = math.atan2(self.player_pos.y - y, self.player_pos.x - x)
            points = []
            for i in range(3):
                angle = angle_to_player + (i * 2 * math.pi / 3)
                points.append((x + self.SPIDER_RADIUS * math.cos(angle), y + self.SPIDER_RADIUS * math.sin(angle)))
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_SPIDER)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_SPIDER)

        # Player attack AoE
        if self.attack_timer > 0:
            alpha = int(200 * (self.attack_timer / self.PLAYER_ATTACK_DURATION))
            radius = int(self.PLAYER_ATTACK_RADIUS * (1 - self.attack_timer / self.PLAYER_ATTACK_DURATION))
            color = (*self.COLOR_ATTACK, alpha)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (self.player_pos.x - radius, self.player_pos.y - radius))

        # Player
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        if self.last_hit_timer > 0 and (self.last_hit_timer // 2) % 2 == 0:
            # Flicker when hit
            pass
        else:
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 6, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_BG)

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Screen flash on hit
        if self.last_hit_timer == (self.FPS // 2) - 1:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((255, 0, 0, 80))
            self.screen.blit(s, (0, 0))

    def _render_ui(self):
        # Health bar
        health_pct = max(0, self.player_health / 100)
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_pct), bar_height))
        health_text = self.font_small.render(f"HP: {int(self.player_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10 + bar_width + 10, 12))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Serum Count
        serum_text = self.font_large.render(f"SERUM: {self.serum_collected}/{self.SERUM_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(serum_text, (self.WIDTH - serum_text.get_width() - 10, self.HEIGHT - serum_text.get_height() - 10))
        
        if self.game_over:
            outcome_text_str = "VICTORY" if self.serum_collected >= self.SERUM_COUNT else "GAME OVER"
            outcome_text = self.font_large.render(outcome_text_str, True, self.COLOR_PLAYER if outcome_text_str == "VICTORY" else self.COLOR_SPIDER)
            text_rect = outcome_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "serum_collected": self.serum_collected,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to see the game window
    render_mode = "human" 
    
    if render_mode == "human":
        GameEnv.metadata["render_modes"].append("human")
        GameEnv.render = lambda self, mode='human': self._render_human()
        GameEnv._render_human = lambda self: pygame.display.update(pygame.transform.scale(self.screen, (self.WIDTH * 2, self.HEIGHT * 2)))
        
        # Monkey-patch __init__ to create a display
        original_init = GameEnv.__init__
        def new_init(self, render_mode="human"):
            original_init(self, render_mode)
            if render_mode == "human":
                self.window = pygame.display.set_mode((self.WIDTH * 2, self.HEIGHT * 2))
                pygame.display.set_caption("Spider Survival")
        GameEnv.__init__ = new_init

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()

    terminated = False
    total_reward = 0
    
    # --- Manual Play Controls ---
    # This block allows a human to play the game
    if render_mode == 'human':
        print(GameEnv.user_guide)
        while not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if render_mode == "human":
                env.render()
                env.clock.tick(env.FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    # --- RL Agent Loop Example ---
    else: 
        for _ in range(1000):
            action = env.action_space.sample()  # Replace with your agent's action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
                obs, info = env.reset()
                total_reward = 0

    env.close()